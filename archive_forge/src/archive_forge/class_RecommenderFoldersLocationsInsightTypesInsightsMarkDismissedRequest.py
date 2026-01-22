from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderFoldersLocationsInsightTypesInsightsMarkDismissedRequest(_messages.Message):
    """A RecommenderFoldersLocationsInsightTypesInsightsMarkDismissedRequest
  object.

  Fields:
    googleCloudRecommenderV1alpha2MarkInsightDismissedRequest: A
      GoogleCloudRecommenderV1alpha2MarkInsightDismissedRequest resource to be
      passed as the request body.
    name: Name of the insight.
  """
    googleCloudRecommenderV1alpha2MarkInsightDismissedRequest = _messages.MessageField('GoogleCloudRecommenderV1alpha2MarkInsightDismissedRequest', 1)
    name = _messages.StringField(2, required=True)