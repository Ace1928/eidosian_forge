from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2ListRecommendersResponse(_messages.Message):
    """Response for the `ListRecommender` method. Next ID: 3

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    recommenders: The set of recommenders available
  """
    nextPageToken = _messages.StringField(1)
    recommenders = _messages.MessageField('GoogleCloudRecommenderV1alpha2RecommenderType', 2, repeated=True)