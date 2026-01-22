from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListFeatureViewsResponse(_messages.Message):
    """Response message for FeatureOnlineStoreAdminService.ListFeatureViews.

  Fields:
    featureViews: The FeatureViews matching the request.
    nextPageToken: A token, which can be sent as
      ListFeatureViewsRequest.page_token to retrieve the next page. If this
      field is omitted, there are no subsequent pages.
  """
    featureViews = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureView', 1, repeated=True)
    nextPageToken = _messages.StringField(2)