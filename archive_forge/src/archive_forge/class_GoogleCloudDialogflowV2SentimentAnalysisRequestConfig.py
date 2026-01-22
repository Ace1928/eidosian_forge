from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SentimentAnalysisRequestConfig(_messages.Message):
    """Configures the types of sentiment analysis to perform.

  Fields:
    analyzeQueryTextSentiment: Instructs the service to perform sentiment
      analysis on `query_text`. If not provided, sentiment analysis is not
      performed on `query_text`.
  """
    analyzeQueryTextSentiment = _messages.BooleanField(1)