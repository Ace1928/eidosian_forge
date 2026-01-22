from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1beta1ListInsightTypesResponse(_messages.Message):
    """Response for the `ListInsightTypes` method. Next ID: 3

  Fields:
    insightTypes: The set of recommenders available
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    insightTypes = _messages.MessageField('GoogleCloudRecommenderV1beta1InsightType', 1, repeated=True)
    nextPageToken = _messages.StringField(2)