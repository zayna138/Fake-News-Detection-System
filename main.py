import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading dataset...")

# Load dataset
df = pd.read_csv("Fake.csv")

# Keep only needed columns
df = df[['text']]

# Create labels
# Since dataset contains fake news,
# label everything as 0
df['label'] = 0

# Duplicate data for demo
real_df = df.copy()

# Change labels to REAL
real_df['label'] = 1

# Combine fake + real
data = pd.concat([df, real_df])

# Shuffle dataset
data = data.sample(frac=1)

# Inputs and outputs
x = data['text']

y = data['label']

# Convert text into numbers
vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2
)

# Train model
model = LogisticRegression()

model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Test News
news = ["Government announces new education policy"]

news_vector = vectorizer.transform(news)

prediction = model.predict(news_vector)

if prediction[0] == 1:
    print("REAL NEWS")
else:
    print("FAKE NEWS")