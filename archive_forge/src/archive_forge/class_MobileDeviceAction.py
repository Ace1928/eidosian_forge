from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MobileDeviceAction(_messages.Message):
    """JSON request template for firing commands on Mobile Device in Directory

  Devices API.

  Fields:
    action: Action to be taken on the Mobile Device
  """
    action = _messages.StringField(1)