from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTaskDeleteRequest(_messages.Message):
    """A FusiontablesTaskDeleteRequest object.

  Fields:
    tableId: Table from which the task is being deleted.
    taskId: A string attribute.
  """
    tableId = _messages.StringField(1, required=True)
    taskId = _messages.StringField(2, required=True)