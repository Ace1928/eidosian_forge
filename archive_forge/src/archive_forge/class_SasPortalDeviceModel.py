from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalDeviceModel(_messages.Message):
    """Information about the model of the device.

  Fields:
    firmwareVersion: The firmware version of the device.
    hardwareVersion: The hardware version of the device.
    name: The name of the device model.
    softwareVersion: The software version of the device.
    vendor: The name of the device vendor.
  """
    firmwareVersion = _messages.StringField(1)
    hardwareVersion = _messages.StringField(2)
    name = _messages.StringField(3)
    softwareVersion = _messages.StringField(4)
    vendor = _messages.StringField(5)