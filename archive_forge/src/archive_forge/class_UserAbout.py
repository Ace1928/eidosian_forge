from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserAbout(_messages.Message):
    """JSON template for About (notes) of a user in Directory API.

  Fields:
    contentType: About entry can have a type which indicates the content type.
      It can either be plain or html. By default, notes contents are assumed
      to contain plain text.
    value: Actual value of notes.
  """
    contentType = _messages.StringField(1)
    value = _messages.StringField(2)