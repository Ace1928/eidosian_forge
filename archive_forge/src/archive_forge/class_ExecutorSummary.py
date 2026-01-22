from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutorSummary(_messages.Message):
    """Details about executors used by the application.

  Messages:
    AttributesValue: A AttributesValue object.
    ExecutorLogsValue: A ExecutorLogsValue object.
    ResourcesValue: A ResourcesValue object.

  Fields:
    activeTasks: A integer attribute.
    addTime: A string attribute.
    attributes: A AttributesValue attribute.
    completedTasks: A integer attribute.
    diskUsed: A string attribute.
    excludedInStages: A string attribute.
    executorId: A string attribute.
    executorLogs: A ExecutorLogsValue attribute.
    failedTasks: A integer attribute.
    hostPort: A string attribute.
    isActive: A boolean attribute.
    isExcluded: A boolean attribute.
    maxMemory: A string attribute.
    maxTasks: A integer attribute.
    memoryMetrics: A MemoryMetrics attribute.
    memoryUsed: A string attribute.
    peakMemoryMetrics: A ExecutorMetrics attribute.
    rddBlocks: A integer attribute.
    removeReason: A string attribute.
    removeTime: A string attribute.
    resourceProfileId: A integer attribute.
    resources: A ResourcesValue attribute.
    totalCores: A integer attribute.
    totalDurationMillis: A string attribute.
    totalGcTimeMillis: A string attribute.
    totalInputBytes: A string attribute.
    totalShuffleRead: A string attribute.
    totalShuffleWrite: A string attribute.
    totalTasks: A integer attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributesValue(_messages.Message):
        """A AttributesValue object.

    Messages:
      AdditionalProperty: An additional property for a AttributesValue object.

    Fields:
      additionalProperties: Additional properties of type AttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExecutorLogsValue(_messages.Message):
        """A ExecutorLogsValue object.

    Messages:
      AdditionalProperty: An additional property for a ExecutorLogsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExecutorLogsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExecutorLogsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourcesValue(_messages.Message):
        """A ResourcesValue object.

    Messages:
      AdditionalProperty: An additional property for a ResourcesValue object.

    Fields:
      additionalProperties: Additional properties of type ResourcesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A ResourceInformation attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ResourceInformation', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    activeTasks = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    addTime = _messages.StringField(2)
    attributes = _messages.MessageField('AttributesValue', 3)
    completedTasks = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    diskUsed = _messages.IntegerField(5)
    excludedInStages = _messages.IntegerField(6, repeated=True)
    executorId = _messages.StringField(7)
    executorLogs = _messages.MessageField('ExecutorLogsValue', 8)
    failedTasks = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    hostPort = _messages.StringField(10)
    isActive = _messages.BooleanField(11)
    isExcluded = _messages.BooleanField(12)
    maxMemory = _messages.IntegerField(13)
    maxTasks = _messages.IntegerField(14, variant=_messages.Variant.INT32)
    memoryMetrics = _messages.MessageField('MemoryMetrics', 15)
    memoryUsed = _messages.IntegerField(16)
    peakMemoryMetrics = _messages.MessageField('ExecutorMetrics', 17)
    rddBlocks = _messages.IntegerField(18, variant=_messages.Variant.INT32)
    removeReason = _messages.StringField(19)
    removeTime = _messages.StringField(20)
    resourceProfileId = _messages.IntegerField(21, variant=_messages.Variant.INT32)
    resources = _messages.MessageField('ResourcesValue', 22)
    totalCores = _messages.IntegerField(23, variant=_messages.Variant.INT32)
    totalDurationMillis = _messages.IntegerField(24)
    totalGcTimeMillis = _messages.IntegerField(25)
    totalInputBytes = _messages.IntegerField(26)
    totalShuffleRead = _messages.IntegerField(27)
    totalShuffleWrite = _messages.IntegerField(28)
    totalTasks = _messages.IntegerField(29, variant=_messages.Variant.INT32)