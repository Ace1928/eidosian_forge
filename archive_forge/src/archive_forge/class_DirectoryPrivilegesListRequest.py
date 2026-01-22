from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryPrivilegesListRequest(_messages.Message):
    """A DirectoryPrivilegesListRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
  """
    customer = _messages.StringField(1, required=True)