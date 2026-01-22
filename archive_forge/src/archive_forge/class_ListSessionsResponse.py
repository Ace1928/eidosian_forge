from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSessionsResponse(_messages.Message):
    """The response for ListSessions.

  Fields:
    nextPageToken: `next_page_token` can be sent in a subsequent ListSessions
      call to fetch more of the matching sessions.
    sessions: The list of requested sessions.
  """
    nextPageToken = _messages.StringField(1)
    sessions = _messages.MessageField('Session', 2, repeated=True)