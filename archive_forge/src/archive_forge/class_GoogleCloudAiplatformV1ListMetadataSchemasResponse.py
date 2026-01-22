from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListMetadataSchemasResponse(_messages.Message):
    """Response message for MetadataService.ListMetadataSchemas.

  Fields:
    metadataSchemas: The MetadataSchemas found for the MetadataStore.
    nextPageToken: A token, which can be sent as
      ListMetadataSchemasRequest.page_token to retrieve the next page. If this
      field is not populated, there are no subsequent pages.
  """
    metadataSchemas = _messages.MessageField('GoogleCloudAiplatformV1MetadataSchema', 1, repeated=True)
    nextPageToken = _messages.StringField(2)