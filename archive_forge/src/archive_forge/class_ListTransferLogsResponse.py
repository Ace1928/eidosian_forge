from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTransferLogsResponse(_messages.Message):
    """The returned list transfer run messages.

  Fields:
    nextPageToken: Output only. The next-pagination token. For multiple-page
      list results, this token can be used as the
      `GetTransferRunLogRequest.page_token` to request the next page of list
      results.
    transferMessages: Output only. The stored pipeline transfer messages.
  """
    nextPageToken = _messages.StringField(1)
    transferMessages = _messages.MessageField('TransferMessage', 2, repeated=True)