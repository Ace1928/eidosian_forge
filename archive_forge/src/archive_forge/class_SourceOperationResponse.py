from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceOperationResponse(_messages.Message):
    """The result of a SourceOperationRequest, specified in
  ReportWorkItemStatusRequest.source_operation when the work item is
  completed.

  Fields:
    getMetadata: A response to a request to get metadata about a source.
    split: A response to a request to split a source.
  """
    getMetadata = _messages.MessageField('SourceGetMetadataResponse', 1)
    split = _messages.MessageField('SourceSplitResponse', 2)