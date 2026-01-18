from transformers import pipeline


def summarize_text(text):
    # Load the summarization pipeline
    summarizer = pipeline("summarization")

    # Perform summarization
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    # Extract and return the summary text
    return summary[0]["summary_text"]


# Example text
text = """
    The Transformers library provides state-of-the-art machine learning algorithms like BERT, GPT-2, T5, and others. 
    They are designed to handle various tasks like translation, summarization, and more. The library is built with a focus on 
    easy integration and high performance in various machine learning contexts.
    """

# Get the summary
summary_text = summarize_text(text)
print("Summary:", summary_text)
