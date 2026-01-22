from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDatabasesResponse(_messages.Message):
    """The response for ListDatabases.

  Fields:
    databases: Databases that matched the request.
    nextPageToken: `next_page_token` can be sent in a subsequent ListDatabases
      call to fetch more of the matching databases.
  """
    databases = _messages.MessageField('Database', 1, repeated=True)
    nextPageToken = _messages.StringField(2)