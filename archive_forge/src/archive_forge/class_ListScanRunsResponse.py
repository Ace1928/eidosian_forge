from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListScanRunsResponse(_messages.Message):
    """Response for the `ListScanRuns` method.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    scanRuns: The list of ScanRuns returned.
  """
    nextPageToken = _messages.StringField(1)
    scanRuns = _messages.MessageField('ScanRun', 2, repeated=True)