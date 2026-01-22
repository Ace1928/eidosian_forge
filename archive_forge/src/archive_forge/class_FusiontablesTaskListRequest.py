from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTaskListRequest(_messages.Message):
    """A FusiontablesTaskListRequest object.

  Fields:
    maxResults: Maximum number of columns to return. Optional. Default is 5.
    pageToken: A string attribute.
    startIndex: A integer attribute.
    tableId: Table whose tasks are being listed.
  """
    maxResults = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(2)
    startIndex = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    tableId = _messages.StringField(4, required=True)