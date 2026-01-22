from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchGitRefsResponse(_messages.Message):
    """Response for fetching git refs

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    refNames: Name of the refs fetched.
  """
    nextPageToken = _messages.StringField(1)
    refNames = _messages.StringField(2, repeated=True)