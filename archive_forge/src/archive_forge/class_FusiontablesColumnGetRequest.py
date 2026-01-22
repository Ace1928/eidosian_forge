from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesColumnGetRequest(_messages.Message):
    """A FusiontablesColumnGetRequest object.

  Fields:
    columnId: Name or identifier for the column that is being requested.
    tableId: Table to which the column belongs.
  """
    columnId = _messages.StringField(1, required=True)
    tableId = _messages.StringField(2, required=True)