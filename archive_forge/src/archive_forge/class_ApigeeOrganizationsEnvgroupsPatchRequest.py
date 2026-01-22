from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvgroupsPatchRequest(_messages.Message):
    """A ApigeeOrganizationsEnvgroupsPatchRequest object.

  Fields:
    googleCloudApigeeV1EnvironmentGroup: A GoogleCloudApigeeV1EnvironmentGroup
      resource to be passed as the request body.
    name: Required. Name of the environment group to update in the format:
      `organizations/{org}/envgroups/{envgroup}.
    updateMask: Optional. List of fields to be updated.
  """
    googleCloudApigeeV1EnvironmentGroup = _messages.MessageField('GoogleCloudApigeeV1EnvironmentGroup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)