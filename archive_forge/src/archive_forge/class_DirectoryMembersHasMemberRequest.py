from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryMembersHasMemberRequest(_messages.Message):
    """A DirectoryMembersHasMemberRequest object.

  Fields:
    groupKey: Identifies the group in the API request. The value can be the
      group's email address, group alias, or the unique group ID.
    memberKey: Identifies the user member in the API request. The value can be
      the user's primary email address, alias, or unique ID.
  """
    groupKey = _messages.StringField(1, required=True)
    memberKey = _messages.StringField(2, required=True)