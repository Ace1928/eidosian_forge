from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTableDeleteRequest(_messages.Message):
    """A FusiontablesTableDeleteRequest object.

  Fields:
    tableId: ID of the table that is being deleted.
  """
    tableId = _messages.StringField(1, required=True)