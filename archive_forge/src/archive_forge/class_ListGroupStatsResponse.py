from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListGroupStatsResponse(_messages.Message):
    """Contains a set of requested error group stats.

  Fields:
    errorGroupStats: The error group stats which match the given request.
    nextPageToken: If non-empty, more results are available. Pass this token,
      along with the same query parameters as the first request, to view the
      next page of results.
    timeRangeBegin: The timestamp specifies the start time to which the
      request was restricted. The start time is set based on the requested
      time range. It may be adjusted to a later time if a project has exceeded
      the storage quota and older data has been deleted.
  """
    errorGroupStats = _messages.MessageField('ErrorGroupStats', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    timeRangeBegin = _messages.StringField(3)