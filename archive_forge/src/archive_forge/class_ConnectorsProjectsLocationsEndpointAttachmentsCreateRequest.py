from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsEndpointAttachmentsCreateRequest(_messages.Message):
    """A ConnectorsProjectsLocationsEndpointAttachmentsCreateRequest object.

  Fields:
    endpointAttachment: A EndpointAttachment resource to be passed as the
      request body.
    endpointAttachmentId: Required. Identifier to assign to the
      EndpointAttachment. Must be unique within scope of the parent resource.
    parent: Required. Parent resource of the EndpointAttachment, of the form:
      `projects/*/locations/*`
  """
    endpointAttachment = _messages.MessageField('EndpointAttachment', 1)
    endpointAttachmentId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)