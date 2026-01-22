from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesAttachmentsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesAttachmentsDeleteRequest object.

  Fields:
    name: Required. Name of the attachment. Use the following structure in
      your request:
      `organizations/{org}/instances/{instance}/attachments/{attachment}`.
  """
    name = _messages.StringField(1, required=True)