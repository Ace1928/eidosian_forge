from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListEndpointAttachmentsResponse(_messages.Message):
    """Response for ListEndpointAttachments method.

  Fields:
    endpointAttachments: Endpoint attachments in the specified organization.
    nextPageToken: Page token that you can include in an
      `ListEndpointAttachments` request to retrieve the next page. If omitted,
      no subsequent pages exist.
  """
    endpointAttachments = _messages.MessageField('GoogleCloudApigeeV1EndpointAttachment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)