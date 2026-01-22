from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsAppsUpdateRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsAppsUpdateRequest object.

  Fields:
    action: Approve or revoke the consumer key by setting this value to
      `approve` or `revoke`. The `Content-Type` header must be set to
      `application/octet-stream`, with empty body.
    googleCloudApigeeV1AppGroupApp: A GoogleCloudApigeeV1AppGroupApp resource
      to be passed as the request body.
    name: Required. Name of the AppGroup app. Use the following structure in
      your request:
      `organizations/{org}/appgroups/{app_group_name}/apps/{app}`
  """
    action = _messages.StringField(1)
    googleCloudApigeeV1AppGroupApp = _messages.MessageField('GoogleCloudApigeeV1AppGroupApp', 2)
    name = _messages.StringField(3, required=True)