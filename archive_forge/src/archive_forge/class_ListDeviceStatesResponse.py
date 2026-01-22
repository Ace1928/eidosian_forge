from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDeviceStatesResponse(_messages.Message):
    """Response for `ListDeviceStates`.

  Fields:
    deviceStates: The last few device states. States are listed in descending
      order of server update time, starting from the most recent one.
  """
    deviceStates = _messages.MessageField('DeviceState', 1, repeated=True)