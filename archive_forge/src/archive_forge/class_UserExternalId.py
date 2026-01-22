from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserExternalId(_messages.Message):
    """JSON template for an externalId entry.

  Fields:
    customType: Custom type.
    type: The type of the Id.
    value: The value of the id.
  """
    customType = _messages.StringField(1)
    type = _messages.StringField(2)
    value = _messages.StringField(3)