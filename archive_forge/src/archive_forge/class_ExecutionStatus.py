from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionStatus(_messages.Message):
    """ExecutionStatus represents the current state of an Execution.

  Fields:
    cancelledCount: Optional. The number of tasks which reached phase
      Cancelled.
    completionTime: Optional. Represents the time that the execution was
      completed. It is not guaranteed to be set in happens-before order across
      separate operations. It is represented in RFC3339 form and is in UTC.
      +optional
    conditions: Optional. Conditions communicate information about
      ongoing/complete reconciliation processes that bring the "spec" inline
      with the observed state of the world. Execution-specific conditions
      include: * `ResourcesAvailable`: `True` when underlying resources have
      been provisioned. * `Started`: `True` when the execution has started to
      execute. * `Completed`: `True` when the execution has succeeded. `False`
      when the execution has failed.
    failedCount: Optional. The number of tasks which reached phase Failed.
    logUri: Optional. URI where logs for this execution can be found in Cloud
      Console.
    observedGeneration: Optional. The 'generation' of the execution that was
      last processed by the controller.
    retriedCount: Optional. The number of tasks which have retried at least
      once.
    runningCount: Optional. The number of actively running tasks.
    startTime: Optional. Represents the time that the execution started to
      run. It is not guaranteed to be set in happens-before order across
      separate operations. It is represented in RFC3339 form and is in UTC.
    succeededCount: Optional. The number of tasks which reached phase
      Succeeded.
  """
    cancelledCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    completionTime = _messages.StringField(2)
    conditions = _messages.MessageField('GoogleCloudRunV1Condition', 3, repeated=True)
    failedCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    logUri = _messages.StringField(5)
    observedGeneration = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    retriedCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    runningCount = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    startTime = _messages.StringField(9)
    succeededCount = _messages.IntegerField(10, variant=_messages.Variant.INT32)