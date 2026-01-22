from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class TableList(_messages.Message):
    """Represents a list of tables.

  Fields:
    items: List of all requested tables.
    kind: Type name: a list of all tables.
    nextPageToken: Token used to access the next page of this result. No token
      is displayed if there are no more pages left.
  """
    items = _messages.MessageField('Table', 1, repeated=True)
    kind = _messages.StringField(2, default=u'fusiontables#tableList')
    nextPageToken = _messages.StringField(3)