from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsCreateRequest object.

  Fields:
    googleCloudApigeeV1Environment: A GoogleCloudApigeeV1Environment resource
      to be passed as the request body.
    name: Optional. Name of the environment.
    parent: Required. Name of the organization in which the environment will
      be created. Use the following structure in your request:
      `organizations/{org}`
  """
    googleCloudApigeeV1Environment = _messages.MessageField('GoogleCloudApigeeV1Environment', 1)
    name = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)