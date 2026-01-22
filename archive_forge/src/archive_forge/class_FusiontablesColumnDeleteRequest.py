from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesColumnDeleteRequest(_messages.Message):
    """A FusiontablesColumnDeleteRequest object.

  Fields:
    columnId: Name or identifier for the column being deleted.
    tableId: Table from which the column is being deleted.
  """
    columnId = _messages.StringField(1, required=True)
    tableId = _messages.StringField(2, required=True)