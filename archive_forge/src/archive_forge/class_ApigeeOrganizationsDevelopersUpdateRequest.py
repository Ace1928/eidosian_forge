from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersUpdateRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersUpdateRequest object.

  Fields:
    googleCloudApigeeV1Developer: A GoogleCloudApigeeV1Developer resource to
      be passed as the request body.
    name: Required. Email address of the developer. Use the following
      structure in your request:
      `organizations/{org}/developers/{developer_email}`
  """
    googleCloudApigeeV1Developer = _messages.MessageField('GoogleCloudApigeeV1Developer', 1)
    name = _messages.StringField(2, required=True)