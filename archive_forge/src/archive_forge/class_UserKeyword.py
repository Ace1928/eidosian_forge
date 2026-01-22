from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserKeyword(_messages.Message):
    """JSON template for a keyword entry.

  Fields:
    customType: Custom Type.
    type: Each entry can have a type which indicates standard type of that
      entry. For example, keyword could be of type occupation or outlook. In
      addition to the standard type, an entry can have a custom type and can
      give it any name. Such types should have the CUSTOM value as type and
      also have a customType value.
    value: Keyword.
  """
    customType = _messages.StringField(1)
    type = _messages.StringField(2)
    value = _messages.StringField(3)