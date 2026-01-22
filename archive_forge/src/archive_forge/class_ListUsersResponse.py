from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUsersResponse(_messages.Message):
    """Message for response to listing Users

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
    users: The list of User
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    users = _messages.MessageField('User', 3, repeated=True)