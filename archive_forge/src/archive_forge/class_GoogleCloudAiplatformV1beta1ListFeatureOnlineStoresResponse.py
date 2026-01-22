from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListFeatureOnlineStoresResponse(_messages.Message):
    """Response message for
  FeatureOnlineStoreAdminService.ListFeatureOnlineStores.

  Fields:
    featureOnlineStores: The FeatureOnlineStores matching the request.
    nextPageToken: A token, which can be sent as
      ListFeatureOnlineStoresRequest.page_token to retrieve the next page. If
      this field is omitted, there are no subsequent pages.
  """
    featureOnlineStores = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureOnlineStore', 1, repeated=True)
    nextPageToken = _messages.StringField(2)