from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class TaskList(_messages.Message):
    """Represents a list of tasks for a table.

  Fields:
    items: List of all requested tasks.
    kind: Type of the resource. This is always "fusiontables#taskList".
    nextPageToken: Token used to access the next page of this result. No token
      is displayed if there are no more pages left.
    totalItems: Total number of tasks for the table.
  """
    items = _messages.MessageField('Task', 1, repeated=True)
    kind = _messages.StringField(2, default=u'fusiontables#taskList')
    nextPageToken = _messages.StringField(3)
    totalItems = _messages.IntegerField(4, variant=_messages.Variant.INT32)