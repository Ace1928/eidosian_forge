from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobData(_messages.Message):
    """Data corresponding to a spark job.

  Enums:
    StatusValueValuesEnum:

  Messages:
    KillTasksSummaryValue: A KillTasksSummaryValue object.

  Fields:
    completionTime: A string attribute.
    description: A string attribute.
    jobGroup: A string attribute.
    jobId: A string attribute.
    killTasksSummary: A KillTasksSummaryValue attribute.
    name: A string attribute.
    numActiveStages: A integer attribute.
    numActiveTasks: A integer attribute.
    numCompletedIndices: A integer attribute.
    numCompletedStages: A integer attribute.
    numCompletedTasks: A integer attribute.
    numFailedStages: A integer attribute.
    numFailedTasks: A integer attribute.
    numKilledTasks: A integer attribute.
    numSkippedStages: A integer attribute.
    numSkippedTasks: A integer attribute.
    numTasks: A integer attribute.
    skippedStages: A integer attribute.
    sqlExecutionId: A string attribute.
    stageIds: A string attribute.
    status: A StatusValueValuesEnum attribute.
    submissionTime: A string attribute.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """StatusValueValuesEnum enum type.

    Values:
      JOB_EXECUTION_STATUS_UNSPECIFIED: <no description>
      JOB_EXECUTION_STATUS_RUNNING: <no description>
      JOB_EXECUTION_STATUS_SUCCEEDED: <no description>
      JOB_EXECUTION_STATUS_FAILED: <no description>
      JOB_EXECUTION_STATUS_UNKNOWN: <no description>
    """
        JOB_EXECUTION_STATUS_UNSPECIFIED = 0
        JOB_EXECUTION_STATUS_RUNNING = 1
        JOB_EXECUTION_STATUS_SUCCEEDED = 2
        JOB_EXECUTION_STATUS_FAILED = 3
        JOB_EXECUTION_STATUS_UNKNOWN = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class KillTasksSummaryValue(_messages.Message):
        """A KillTasksSummaryValue object.

    Messages:
      AdditionalProperty: An additional property for a KillTasksSummaryValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        KillTasksSummaryValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a KillTasksSummaryValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    completionTime = _messages.StringField(1)
    description = _messages.StringField(2)
    jobGroup = _messages.StringField(3)
    jobId = _messages.IntegerField(4)
    killTasksSummary = _messages.MessageField('KillTasksSummaryValue', 5)
    name = _messages.StringField(6)
    numActiveStages = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    numActiveTasks = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    numCompletedIndices = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    numCompletedStages = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    numCompletedTasks = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    numFailedStages = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    numFailedTasks = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    numKilledTasks = _messages.IntegerField(14, variant=_messages.Variant.INT32)
    numSkippedStages = _messages.IntegerField(15, variant=_messages.Variant.INT32)
    numSkippedTasks = _messages.IntegerField(16, variant=_messages.Variant.INT32)
    numTasks = _messages.IntegerField(17, variant=_messages.Variant.INT32)
    skippedStages = _messages.IntegerField(18, repeated=True, variant=_messages.Variant.INT32)
    sqlExecutionId = _messages.IntegerField(19)
    stageIds = _messages.IntegerField(20, repeated=True)
    status = _messages.EnumField('StatusValueValuesEnum', 21)
    submissionTime = _messages.StringField(22)