from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskMetrics(_messages.Message):
    """Executor Task Metrics

  Fields:
    diskBytesSpilled: A string attribute.
    executorCpuTimeNanos: A string attribute.
    executorDeserializeCpuTimeNanos: A string attribute.
    executorDeserializeTimeMillis: A string attribute.
    executorRunTimeMillis: A string attribute.
    inputMetrics: A InputMetrics attribute.
    jvmGcTimeMillis: A string attribute.
    memoryBytesSpilled: A string attribute.
    outputMetrics: A OutputMetrics attribute.
    peakExecutionMemoryBytes: A string attribute.
    resultSerializationTimeMillis: A string attribute.
    resultSize: A string attribute.
    shuffleReadMetrics: A ShuffleReadMetrics attribute.
    shuffleWriteMetrics: A ShuffleWriteMetrics attribute.
  """
    diskBytesSpilled = _messages.IntegerField(1)
    executorCpuTimeNanos = _messages.IntegerField(2)
    executorDeserializeCpuTimeNanos = _messages.IntegerField(3)
    executorDeserializeTimeMillis = _messages.IntegerField(4)
    executorRunTimeMillis = _messages.IntegerField(5)
    inputMetrics = _messages.MessageField('InputMetrics', 6)
    jvmGcTimeMillis = _messages.IntegerField(7)
    memoryBytesSpilled = _messages.IntegerField(8)
    outputMetrics = _messages.MessageField('OutputMetrics', 9)
    peakExecutionMemoryBytes = _messages.IntegerField(10)
    resultSerializationTimeMillis = _messages.IntegerField(11)
    resultSize = _messages.IntegerField(12)
    shuffleReadMetrics = _messages.MessageField('ShuffleReadMetrics', 13)
    shuffleWriteMetrics = _messages.MessageField('ShuffleWriteMetrics', 14)