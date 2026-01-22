from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeSentimentResponse(_messages.Message):
    """The sentiment analysis response message.

  Fields:
    documentSentiment: The overall sentiment of the input document.
    language: The language of the text, which will be the same as the language
      specified in the request or, if not specified, the automatically-
      detected language. See Document.language field for more details.
    sentences: The sentiment for all the sentences in the document.
  """
    documentSentiment = _messages.MessageField('Sentiment', 1)
    language = _messages.StringField(2)
    sentences = _messages.MessageField('Sentence', 3, repeated=True)