from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StageData(_messages.Message):
    """Data corresponding to a stage.

  Enums:
    StatusValueValuesEnum:

  Messages:
    ExecutorSummaryValue: A ExecutorSummaryValue object.
    KilledTasksSummaryValue: A KilledTasksSummaryValue object.
    LocalityValue: A LocalityValue object.
    TasksValue: A TasksValue object.

  Fields:
    accumulatorUpdates: A AccumulableInfo attribute.
    completionTime: A string attribute.
    description: A string attribute.
    details: A string attribute.
    executorMetricsDistributions: A ExecutorMetricsDistributions attribute.
    executorSummary: A ExecutorSummaryValue attribute.
    failureReason: A string attribute.
    firstTaskLaunchedTime: A string attribute.
    isShufflePushEnabled: A boolean attribute.
    jobIds: A string attribute.
    killedTasksSummary: A KilledTasksSummaryValue attribute.
    locality: A LocalityValue attribute.
    name: A string attribute.
    numActiveTasks: A integer attribute.
    numCompleteTasks: A integer attribute.
    numCompletedIndices: A integer attribute.
    numFailedTasks: A integer attribute.
    numKilledTasks: A integer attribute.
    numTasks: A integer attribute.
    parentStageIds: A string attribute.
    peakExecutorMetrics: A ExecutorMetrics attribute.
    rddIds: A string attribute.
    resourceProfileId: A integer attribute.
    schedulingPool: A string attribute.
    shuffleMergersCount: A integer attribute.
    speculationSummary: A SpeculationStageSummary attribute.
    stageAttemptId: A integer attribute.
    stageId: A string attribute.
    stageMetrics: A StageMetrics attribute.
    status: A StatusValueValuesEnum attribute.
    submissionTime: A string attribute.
    taskQuantileMetrics: Summary metrics fields. These are included in
      response only if present in summary_metrics_mask field in request
    tasks: A TasksValue attribute.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """StatusValueValuesEnum enum type.

    Values:
      STAGE_STATUS_UNSPECIFIED: <no description>
      STAGE_STATUS_ACTIVE: <no description>
      STAGE_STATUS_COMPLETE: <no description>
      STAGE_STATUS_FAILED: <no description>
      STAGE_STATUS_PENDING: <no description>
      STAGE_STATUS_SKIPPED: <no description>
    """
        STAGE_STATUS_UNSPECIFIED = 0
        STAGE_STATUS_ACTIVE = 1
        STAGE_STATUS_COMPLETE = 2
        STAGE_STATUS_FAILED = 3
        STAGE_STATUS_PENDING = 4
        STAGE_STATUS_SKIPPED = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExecutorSummaryValue(_messages.Message):
        """A ExecutorSummaryValue object.

    Messages:
      AdditionalProperty: An additional property for a ExecutorSummaryValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExecutorSummaryValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExecutorSummaryValue object.

      Fields:
        key: Name of the additional property.
        value: A ExecutorStageSummary attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ExecutorStageSummary', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class KilledTasksSummaryValue(_messages.Message):
        """A KilledTasksSummaryValue object.

    Messages:
      AdditionalProperty: An additional property for a KilledTasksSummaryValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        KilledTasksSummaryValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a KilledTasksSummaryValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LocalityValue(_messages.Message):
        """A LocalityValue object.

    Messages:
      AdditionalProperty: An additional property for a LocalityValue object.

    Fields:
      additionalProperties: Additional properties of type LocalityValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LocalityValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TasksValue(_messages.Message):
        """A TasksValue object.

    Messages:
      AdditionalProperty: An additional property for a TasksValue object.

    Fields:
      additionalProperties: Additional properties of type TasksValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TasksValue object.

      Fields:
        key: Name of the additional property.
        value: A TaskData attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('TaskData', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accumulatorUpdates = _messages.MessageField('AccumulableInfo', 1, repeated=True)
    completionTime = _messages.StringField(2)
    description = _messages.StringField(3)
    details = _messages.StringField(4)
    executorMetricsDistributions = _messages.MessageField('ExecutorMetricsDistributions', 5)
    executorSummary = _messages.MessageField('ExecutorSummaryValue', 6)
    failureReason = _messages.StringField(7)
    firstTaskLaunchedTime = _messages.StringField(8)
    isShufflePushEnabled = _messages.BooleanField(9)
    jobIds = _messages.IntegerField(10, repeated=True)
    killedTasksSummary = _messages.MessageField('KilledTasksSummaryValue', 11)
    locality = _messages.MessageField('LocalityValue', 12)
    name = _messages.StringField(13)
    numActiveTasks = _messages.IntegerField(14, variant=_messages.Variant.INT32)
    numCompleteTasks = _messages.IntegerField(15, variant=_messages.Variant.INT32)
    numCompletedIndices = _messages.IntegerField(16, variant=_messages.Variant.INT32)
    numFailedTasks = _messages.IntegerField(17, variant=_messages.Variant.INT32)
    numKilledTasks = _messages.IntegerField(18, variant=_messages.Variant.INT32)
    numTasks = _messages.IntegerField(19, variant=_messages.Variant.INT32)
    parentStageIds = _messages.IntegerField(20, repeated=True)
    peakExecutorMetrics = _messages.MessageField('ExecutorMetrics', 21)
    rddIds = _messages.IntegerField(22, repeated=True)
    resourceProfileId = _messages.IntegerField(23, variant=_messages.Variant.INT32)
    schedulingPool = _messages.StringField(24)
    shuffleMergersCount = _messages.IntegerField(25, variant=_messages.Variant.INT32)
    speculationSummary = _messages.MessageField('SpeculationStageSummary', 26)
    stageAttemptId = _messages.IntegerField(27, variant=_messages.Variant.INT32)
    stageId = _messages.IntegerField(28)
    stageMetrics = _messages.MessageField('StageMetrics', 29)
    status = _messages.EnumField('StatusValueValuesEnum', 30)
    submissionTime = _messages.StringField(31)
    taskQuantileMetrics = _messages.MessageField('TaskQuantileMetrics', 32)
    tasks = _messages.MessageField('TasksValue', 33)