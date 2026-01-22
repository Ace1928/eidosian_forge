from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEndpointAttachmentsGetRequest(_messages.Message):
    """A ApigeeOrganizationsEndpointAttachmentsGetRequest object.

  Fields:
    name: Required. Name of the endpoint attachment. Use the following
      structure in your request:
      `organizations/{org}/endpointAttachments/{endpoint_attachment}`
  """
    name = _messages.StringField(1, required=True)