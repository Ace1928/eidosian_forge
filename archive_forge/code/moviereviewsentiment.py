from textblob import TextBlob

# Movie reviews
reviews = [
    "The movie was fantastic! The acting was superb and the plot kept me engaged throughout.",
    "I didn't enjoy the movie. The story was predictable and the characters were one-dimensional.",
    "The film had its moments, but overall it was a disappointment. The pacing was slow and the ending was unsatisfying.",
]

# Perform sentiment analysis on each review
for review in reviews:
    blob = TextBlob(review)
    sentiment = blob.sentiment

    print("Review:", review)
    print("Sentiment Polarity:", sentiment.polarity)
    print("Sentiment Subjectivity:", sentiment.subjectivity)
    print("---")
