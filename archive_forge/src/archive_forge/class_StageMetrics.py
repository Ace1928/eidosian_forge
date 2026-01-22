from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StageMetrics(_messages.Message):
    """Stage Level Aggregated Metrics

  Fields:
    diskBytesSpilled: A string attribute.
    executorCpuTimeNanos: A string attribute.
    executorDeserializeCpuTimeNanos: A string attribute.
    executorDeserializeTimeMillis: A string attribute.
    executorRunTimeMillis: A string attribute.
    jvmGcTimeMillis: A string attribute.
    memoryBytesSpilled: A string attribute.
    peakExecutionMemoryBytes: A string attribute.
    resultSerializationTimeMillis: A string attribute.
    resultSize: A string attribute.
    stageInputMetrics: A StageInputMetrics attribute.
    stageOutputMetrics: A StageOutputMetrics attribute.
    stageShuffleReadMetrics: A StageShuffleReadMetrics attribute.
    stageShuffleWriteMetrics: A StageShuffleWriteMetrics attribute.
  """
    diskBytesSpilled = _messages.IntegerField(1)
    executorCpuTimeNanos = _messages.IntegerField(2)
    executorDeserializeCpuTimeNanos = _messages.IntegerField(3)
    executorDeserializeTimeMillis = _messages.IntegerField(4)
    executorRunTimeMillis = _messages.IntegerField(5)
    jvmGcTimeMillis = _messages.IntegerField(6)
    memoryBytesSpilled = _messages.IntegerField(7)
    peakExecutionMemoryBytes = _messages.IntegerField(8)
    resultSerializationTimeMillis = _messages.IntegerField(9)
    resultSize = _messages.IntegerField(10)
    stageInputMetrics = _messages.MessageField('StageInputMetrics', 11)
    stageOutputMetrics = _messages.MessageField('StageOutputMetrics', 12)
    stageShuffleReadMetrics = _messages.MessageField('StageShuffleReadMetrics', 13)
    stageShuffleWriteMetrics = _messages.MessageField('StageShuffleWriteMetrics', 14)