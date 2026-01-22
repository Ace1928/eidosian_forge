from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalSignDeviceRequest(_messages.Message):
    """Request for SignDevice.

  Fields:
    device: Required. The device to sign. The device fields name, fcc_id and
      serial_number must be set. The user_id field must be set.
  """
    device = _messages.MessageField('SasPortalDevice', 1)