from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesCreateRequest object.

  Fields:
    environments: Optional. DEPRECATED: DO NOT USE. List of environments that
      will be attached to the instance during creation.
    googleCloudApigeeV1Instance: A GoogleCloudApigeeV1Instance resource to be
      passed as the request body.
    parent: Required. Name of the organization. Use the following structure in
      your request: `organizations/{org}`.
    runtimeVersion: Optional. Software config version for instance creation.
      runtime_version value can contain only alphanumeric characters and
      hyphens (-) and cannot begin or end with a hyphen.
  """
    environments = _messages.StringField(1, repeated=True)
    googleCloudApigeeV1Instance = _messages.MessageField('GoogleCloudApigeeV1Instance', 2)
    parent = _messages.StringField(3, required=True)
    runtimeVersion = _messages.StringField(4)