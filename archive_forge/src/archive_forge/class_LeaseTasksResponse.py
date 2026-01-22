from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LeaseTasksResponse(_messages.Message):
    """Response message for leasing tasks using LeaseTasks.

  Fields:
    tasks: The leased tasks.
  """
    tasks = _messages.MessageField('Task', 1, repeated=True)