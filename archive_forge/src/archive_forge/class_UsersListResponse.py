from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UsersListResponse(_messages.Message):
    """User list response.

  Fields:
    items: List of user resources in the instance.
    kind: This is always *sql#usersList*.
    nextPageToken: Unused.
  """
    items = _messages.MessageField('User', 1, repeated=True)
    kind = _messages.StringField(2)
    nextPageToken = _messages.StringField(3)