from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BindDeviceToGatewayRequest(_messages.Message):
    """Request for `BindDeviceToGateway`.

  Fields:
    deviceId: Required. The device to associate with the specified gateway.
      The value of `device_id` can be either the device numeric ID or the
      user-defined device identifier.
    gatewayId: Required. The value of `gateway_id` can be either the device
      numeric ID or the user-defined device identifier.
  """
    deviceId = _messages.StringField(1)
    gatewayId = _messages.StringField(2)