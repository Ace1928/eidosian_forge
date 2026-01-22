from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesDeviceUsersApproveRequest(_messages.Message):
    """A CloudidentityDevicesDeviceUsersApproveRequest object.

  Fields:
    googleAppsCloudidentityDevicesV1ApproveDeviceUserRequest: A
      GoogleAppsCloudidentityDevicesV1ApproveDeviceUserRequest resource to be
      passed as the request body.
    name: Required. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the Device
      in format: `devices/{device}/deviceUsers/{device_user}`, where device is
      the unique ID assigned to the Device, and device_user is the unique ID
      assigned to the User.
  """
    googleAppsCloudidentityDevicesV1ApproveDeviceUserRequest = _messages.MessageField('GoogleAppsCloudidentityDevicesV1ApproveDeviceUserRequest', 1)
    name = _messages.StringField(2, required=True)