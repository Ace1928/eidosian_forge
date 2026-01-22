from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsTargetserversCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsTargetserversCreateRequest object.

  Fields:
    googleCloudApigeeV1TargetServer: A GoogleCloudApigeeV1TargetServer
      resource to be passed as the request body.
    name: Optional. The ID to give the TargetServer. This will overwrite the
      value in TargetServer.
    parent: Required. The parent environment name under which the TargetServer
      will be created. Must be of the form
      `organizations/{org}/environments/{env}`.
  """
    googleCloudApigeeV1TargetServer = _messages.MessageField('GoogleCloudApigeeV1TargetServer', 1)
    name = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)