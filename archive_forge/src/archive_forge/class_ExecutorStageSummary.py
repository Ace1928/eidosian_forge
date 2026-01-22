from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutorStageSummary(_messages.Message):
    """Executor resources consumed by a stage.

  Fields:
    diskBytesSpilled: A string attribute.
    executorId: A string attribute.
    failedTasks: A integer attribute.
    inputBytes: A string attribute.
    inputRecords: A string attribute.
    isExcludedForStage: A boolean attribute.
    killedTasks: A integer attribute.
    memoryBytesSpilled: A string attribute.
    outputBytes: A string attribute.
    outputRecords: A string attribute.
    peakMemoryMetrics: A ExecutorMetrics attribute.
    shuffleRead: A string attribute.
    shuffleReadRecords: A string attribute.
    shuffleWrite: A string attribute.
    shuffleWriteRecords: A string attribute.
    stageAttemptId: A integer attribute.
    stageId: A string attribute.
    succeededTasks: A integer attribute.
    taskTimeMillis: A string attribute.
  """
    diskBytesSpilled = _messages.IntegerField(1)
    executorId = _messages.StringField(2)
    failedTasks = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    inputBytes = _messages.IntegerField(4)
    inputRecords = _messages.IntegerField(5)
    isExcludedForStage = _messages.BooleanField(6)
    killedTasks = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    memoryBytesSpilled = _messages.IntegerField(8)
    outputBytes = _messages.IntegerField(9)
    outputRecords = _messages.IntegerField(10)
    peakMemoryMetrics = _messages.MessageField('ExecutorMetrics', 11)
    shuffleRead = _messages.IntegerField(12)
    shuffleReadRecords = _messages.IntegerField(13)
    shuffleWrite = _messages.IntegerField(14)
    shuffleWriteRecords = _messages.IntegerField(15)
    stageAttemptId = _messages.IntegerField(16, variant=_messages.Variant.INT32)
    stageId = _messages.IntegerField(17)
    succeededTasks = _messages.IntegerField(18, variant=_messages.Variant.INT32)
    taskTimeMillis = _messages.IntegerField(19)