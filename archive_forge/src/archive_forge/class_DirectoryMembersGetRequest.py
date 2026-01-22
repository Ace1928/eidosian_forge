from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryMembersGetRequest(_messages.Message):
    """A DirectoryMembersGetRequest object.

  Fields:
    groupKey: Email or immutable ID of the group
    memberKey: Email or immutable ID of the member
  """
    groupKey = _messages.StringField(1, required=True)
    memberKey = _messages.StringField(2, required=True)