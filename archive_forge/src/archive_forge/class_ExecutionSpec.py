from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionSpec(_messages.Message):
    """ExecutionSpec describes how the execution will look.

  Fields:
    parallelism: Optional. Specifies the maximum desired number of tasks the
      execution should run at given time. Must be <= task_count. When the job
      is run, if this field is 0 or unset, the maximum possible value will be
      used for that execution. The actual number of tasks running in steady
      state will be less than this number when there are fewer tasks waiting
      to be completed, i.e. when the work left to do is less than max
      parallelism.
    taskCount: Optional. Specifies the desired number of tasks the execution
      should run. Setting to 1 means that parallelism is limited to 1 and the
      success of that task signals the success of the execution. Defaults to
      1.
    template: Optional. The template used to create tasks for this execution.
  """
    parallelism = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    taskCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    template = _messages.MessageField('TaskTemplateSpec', 3)