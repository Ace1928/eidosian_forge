from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesAttachmentsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesAttachmentsCreateRequest object.

  Fields:
    googleCloudApigeeV1InstanceAttachment: A
      GoogleCloudApigeeV1InstanceAttachment resource to be passed as the
      request body.
    parent: Required. Name of the instance. Use the following structure in
      your request: `organizations/{org}/instances/{instance}`.
  """
    googleCloudApigeeV1InstanceAttachment = _messages.MessageField('GoogleCloudApigeeV1InstanceAttachment', 1)
    parent = _messages.StringField(2, required=True)