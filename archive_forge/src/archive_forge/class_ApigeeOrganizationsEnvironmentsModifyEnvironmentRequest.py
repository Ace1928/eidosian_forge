from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsModifyEnvironmentRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsModifyEnvironmentRequest object.

  Fields:
    googleCloudApigeeV1Environment: A GoogleCloudApigeeV1Environment resource
      to be passed as the request body.
    name: Required. Name of the environment. Use the following structure in
      your request: `organizations/{org}/environments/{environment}`.
    updateMask: List of fields to be updated. Fields that can be updated:
      node_config.
  """
    googleCloudApigeeV1Environment = _messages.MessageField('GoogleCloudApigeeV1Environment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)