from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Overrides(_messages.Message):
    """RunJob Overrides that contains Execution fields to be overridden on the
  go.

  Fields:
    containerOverrides: Per container override specification.
    taskCount: The desired number of tasks the execution should run. Will
      replace existing task_count value.
    timeoutSeconds: Duration in seconds the task may be active before the
      system will actively try to mark it failed and kill associated
      containers. Will replace existing timeout_seconds value.
  """
    containerOverrides = _messages.MessageField('ContainerOverride', 1, repeated=True)
    taskCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    timeoutSeconds = _messages.IntegerField(3, variant=_messages.Variant.INT32)