from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListStepEntriesResponse(_messages.Message):
    """Response message for ExecutionHistory.ListStepEntries.

  Fields:
    nextPageToken: A token to retrieve next page of results. Pass this value
      in the ListStepEntriesRequest.page_token field in the subsequent call to
      `ListStepEntries` method to retrieve the next page of results.
    stepEntries: The list of entries.
    totalSize: Indicates the total number of StepEntries that matched the
      request filter. For running executions, this number shows the number of
      StepEntries that are executed thus far.
  """
    nextPageToken = _messages.StringField(1)
    stepEntries = _messages.MessageField('StepEntry', 2, repeated=True)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)