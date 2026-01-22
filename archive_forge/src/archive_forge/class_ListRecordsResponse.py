from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRecordsResponse(_messages.Message):
    """Message for response to listing Records.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    records: The list of Records.
  """
    nextPageToken = _messages.StringField(1)
    records = _messages.MessageField('Record', 2, repeated=True)