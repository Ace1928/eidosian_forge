from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDeviceConfigVersionsResponse(_messages.Message):
    """Response for `ListDeviceConfigVersions`.

  Fields:
    deviceConfigs: The device configuration for the last few versions.
      Versions are listed in decreasing order, starting from the most recent
      one.
  """
    deviceConfigs = _messages.MessageField('DeviceConfig', 1, repeated=True)