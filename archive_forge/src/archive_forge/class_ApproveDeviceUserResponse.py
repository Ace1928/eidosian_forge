from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApproveDeviceUserResponse(_messages.Message):
    """Response message for approving the device to access user data.

  Fields:
    deviceUser: Resultant DeviceUser object for the action.
  """
    deviceUser = _messages.MessageField('DeviceUser', 1)