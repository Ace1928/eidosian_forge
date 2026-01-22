from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListPublisherModelsResponse(_messages.Message):
    """Response message for ModelGardenService.ListPublisherModels.

  Fields:
    nextPageToken: A token to retrieve next page of results. Pass to
      ListPublisherModels.page_token to obtain that page.
    publisherModels: List of PublisherModels in the requested page.
  """
    nextPageToken = _messages.StringField(1)
    publisherModels = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModel', 2, repeated=True)