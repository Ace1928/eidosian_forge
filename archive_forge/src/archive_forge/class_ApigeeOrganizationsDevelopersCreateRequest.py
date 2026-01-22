from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersCreateRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersCreateRequest object.

  Fields:
    googleCloudApigeeV1Developer: A GoogleCloudApigeeV1Developer resource to
      be passed as the request body.
    parent: Required. Name of the Apigee organization in which the developer
      is created. Use the following structure in your request:
      `organizations/{org}`.
  """
    googleCloudApigeeV1Developer = _messages.MessageField('GoogleCloudApigeeV1Developer', 1)
    parent = _messages.StringField(2, required=True)