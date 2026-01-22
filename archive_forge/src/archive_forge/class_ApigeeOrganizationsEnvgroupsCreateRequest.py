from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvgroupsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvgroupsCreateRequest object.

  Fields:
    googleCloudApigeeV1EnvironmentGroup: A GoogleCloudApigeeV1EnvironmentGroup
      resource to be passed as the request body.
    name: Optional. ID of the environment group. Overrides any ID in the
      environment_group resource.
    parent: Required. Name of the organization in which to create the
      environment group in the following format: `organizations/{org}`.
  """
    googleCloudApigeeV1EnvironmentGroup = _messages.MessageField('GoogleCloudApigeeV1EnvironmentGroup', 1)
    name = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)