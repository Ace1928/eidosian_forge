from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class ColumnList(_messages.Message):
    """Represents a list of columns in a table.

  Fields:
    items: List of all requested columns.
    kind: Type name: a list of all columns.
    nextPageToken: Token used to access the next page of this result. No token
      is displayed if there are no more pages left.
    totalItems: Total number of columns for the table.
  """
    items = _messages.MessageField('Column', 1, repeated=True)
    kind = _messages.StringField(2, default=u'fusiontables#columnList')
    nextPageToken = _messages.StringField(3)
    totalItems = _messages.IntegerField(4, variant=_messages.Variant.INT32)