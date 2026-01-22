from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsEndpointAttachmentsPatchRequest(_messages.Message):
    """A ConnectorsProjectsLocationsEndpointAttachmentsPatchRequest object.

  Fields:
    endpointAttachment: A EndpointAttachment resource to be passed as the
      request body.
    name: Output only. Resource name of the Endpoint Attachment. Format: proje
      cts/{project}/locations/{location}/endpointAttachments/{endpoint_attachm
      ent}
    updateMask: Required. The list of fields to update. Fields are specified
      relative to the endpointAttachment. A field will be overwritten if it is
      in the mask. You can modify only the fields listed below. To update the
      endpointAttachment details: * `description` * `labels`
  """
    endpointAttachment = _messages.MessageField('EndpointAttachment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)