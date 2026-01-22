from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StagePerformanceStandaloneInsight(_messages.Message):
    """Standalone performance insights for a specific stage.

  Fields:
    biEngineReasons: Output only. If present, the stage had the following
      reasons for being disqualified from BI Engine execution.
    highCardinalityJoins: Output only. High cardinality joins in the stage.
    insufficientShuffleQuota: Output only. True if the stage has insufficient
      shuffle quota.
    partitionSkew: Output only. Partition skew in the stage.
    slotContention: Output only. True if the stage has a slot contention
      issue.
    stageId: Output only. The stage id that the insight mapped to.
  """
    biEngineReasons = _messages.MessageField('BiEngineReason', 1, repeated=True)
    highCardinalityJoins = _messages.MessageField('HighCardinalityJoin', 2, repeated=True)
    insufficientShuffleQuota = _messages.BooleanField(3)
    partitionSkew = _messages.MessageField('PartitionSkew', 4)
    slotContention = _messages.BooleanField(5)
    stageId = _messages.IntegerField(6)