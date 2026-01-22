from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChildStatusReference(_messages.Message):
    """ChildStatusReference is used to point to the statuses of individual
  TaskRuns and Runs within this PipelineRun.

  Enums:
    TypeValueValuesEnum: Output only. Type of the child reference.

  Fields:
    name: Name is the name of the TaskRun or Run this is referencing.
    pipelineTaskName: PipelineTaskName is the name of the PipelineTask this is
      referencing.
    type: Output only. Type of the child reference.
    whenExpressions: WhenExpressions is the list of checks guarding the
      execution of the PipelineTask
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. Type of the child reference.

    Values:
      TYPE_UNSPECIFIED: Default enum type; should not be used.
      TASK_RUN: TaskRun.
    """
        TYPE_UNSPECIFIED = 0
        TASK_RUN = 1
    name = _messages.StringField(1)
    pipelineTaskName = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)
    whenExpressions = _messages.MessageField('WhenExpression', 4, repeated=True)