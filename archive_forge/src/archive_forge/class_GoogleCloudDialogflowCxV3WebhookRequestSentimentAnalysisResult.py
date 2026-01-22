from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3WebhookRequestSentimentAnalysisResult(_messages.Message):
    """Represents the result of sentiment analysis.

  Fields:
    magnitude: A non-negative number in the [0, +inf) range, which represents
      the absolute magnitude of sentiment, regardless of score (positive or
      negative).
    score: Sentiment score between -1.0 (negative sentiment) and 1.0 (positive
      sentiment).
  """
    magnitude = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    score = _messages.FloatField(2, variant=_messages.Variant.FLOAT)