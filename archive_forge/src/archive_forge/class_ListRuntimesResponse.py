from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRuntimesResponse(_messages.Message):
    """Response message for Applications.ListRuntimes.

  Fields:
    nextPageToken: Continuation token for fetching the next page of results.
    runtimes: The runtimes available to the requested application.
  """
    nextPageToken = _messages.StringField(1)
    runtimes = _messages.MessageField('Runtime', 2, repeated=True)