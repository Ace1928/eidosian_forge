from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesCancelWipeRequest(_messages.Message):
    """A CloudidentityDevicesCancelWipeRequest object.

  Fields:
    googleAppsCloudidentityDevicesV1CancelWipeDeviceRequest: A
      GoogleAppsCloudidentityDevicesV1CancelWipeDeviceRequest resource to be
      passed as the request body.
    name: Required. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the Device
      in format: `devices/{device}`, where device is the unique ID assigned to
      the Device.
  """
    googleAppsCloudidentityDevicesV1CancelWipeDeviceRequest = _messages.MessageField('GoogleAppsCloudidentityDevicesV1CancelWipeDeviceRequest', 1)
    name = _messages.StringField(2, required=True)