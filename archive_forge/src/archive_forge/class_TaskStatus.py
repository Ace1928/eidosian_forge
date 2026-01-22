from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskStatus(_messages.Message):
    """TaskStatus represents the status of a task.

  Fields:
    completionTime: Optional. Represents time when the task was completed. It
      is not guaranteed to be set in happens-before order across separate
      operations. It is represented in RFC3339 form and is in UTC.
    conditions: Optional. Conditions communicate information about
      ongoing/complete reconciliation processes that bring the "spec" inline
      with the observed state of the world. Task-specific conditions include:
      * `Started`: `True` when the task has started to execute. * `Completed`:
      `True` when the task has succeeded. `False` when the task has failed.
    index: Required. Index of the task, unique per execution, and beginning at
      0.
    lastAttemptResult: Optional. Result of the last attempt of this task.
    logUri: Optional. URI where logs for this task can be found in Cloud
      Console.
    observedGeneration: Optional. The 'generation' of the task that was last
      processed by the controller.
    retried: Optional. The number of times this task was retried. Instances
      are retried when they fail up to the maxRetries limit.
    startTime: Optional. Represents time when the task started to run. It is
      not guaranteed to be set in happens-before order across separate
      operations. It is represented in RFC3339 form and is in UTC.
  """
    completionTime = _messages.StringField(1)
    conditions = _messages.MessageField('GoogleCloudRunV1Condition', 2, repeated=True)
    index = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    lastAttemptResult = _messages.MessageField('TaskAttemptResult', 4)
    logUri = _messages.StringField(5)
    observedGeneration = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    retried = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    startTime = _messages.StringField(8)