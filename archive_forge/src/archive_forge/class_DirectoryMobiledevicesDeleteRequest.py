from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryMobiledevicesDeleteRequest(_messages.Message):
    """A DirectoryMobiledevicesDeleteRequest object.

  Fields:
    customerId: Immutable ID of the G Suite account
    resourceId: Immutable ID of Mobile Device
  """
    customerId = _messages.StringField(1, required=True)
    resourceId = _messages.StringField(2, required=True)