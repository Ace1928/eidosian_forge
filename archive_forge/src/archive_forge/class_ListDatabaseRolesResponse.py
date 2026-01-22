from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDatabaseRolesResponse(_messages.Message):
    """The response for ListDatabaseRoles.

  Fields:
    databaseRoles: Database roles that matched the request.
    nextPageToken: `next_page_token` can be sent in a subsequent
      ListDatabaseRoles call to fetch more of the matching roles.
  """
    databaseRoles = _messages.MessageField('DatabaseRole', 1, repeated=True)
    nextPageToken = _messages.StringField(2)