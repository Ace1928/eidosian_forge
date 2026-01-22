from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListPersistentResourcesResponse(_messages.Message):
    """Response message for PersistentResourceService.ListPersistentResources

  Fields:
    nextPageToken: A token to retrieve next page of results. Pass to
      ListPersistentResourcesRequest.page_token to obtain that page.
    persistentResources: A GoogleCloudAiplatformV1beta1PersistentResource
      attribute.
  """
    nextPageToken = _messages.StringField(1)
    persistentResources = _messages.MessageField('GoogleCloudAiplatformV1beta1PersistentResource', 2, repeated=True)