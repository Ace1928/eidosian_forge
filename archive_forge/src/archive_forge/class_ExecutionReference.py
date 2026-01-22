from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionReference(_messages.Message):
    """Reference to an Execution. Use /Executions.GetExecution with the given
  name to get full execution including the latest status.

  Fields:
    completionTimestamp: Optional. Completion timestamp of the execution.
    creationTimestamp: Optional. Creation timestamp of the execution.
    name: Optional. Name of the execution.
  """
    completionTimestamp = _messages.StringField(1)
    creationTimestamp = _messages.StringField(2)
    name = _messages.StringField(3)