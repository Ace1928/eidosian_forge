from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesColumnListRequest(_messages.Message):
    """A FusiontablesColumnListRequest object.

  Fields:
    maxResults: Maximum number of columns to return. Optional. Default is 5.
    pageToken: Continuation token specifying which result page to return.
      Optional.
    tableId: Table whose columns are being listed.
  """
    maxResults = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(2)
    tableId = _messages.StringField(3, required=True)