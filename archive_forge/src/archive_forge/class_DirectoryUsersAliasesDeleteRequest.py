from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersAliasesDeleteRequest(_messages.Message):
    """A DirectoryUsersAliasesDeleteRequest object.

  Fields:
    alias: The alias to be removed
    userKey: Email or immutable ID of the user
  """
    alias = _messages.StringField(1, required=True)
    userKey = _messages.StringField(2, required=True)