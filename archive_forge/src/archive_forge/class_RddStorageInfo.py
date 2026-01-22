from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RddStorageInfo(_messages.Message):
    """Overall data about RDD storage.

  Fields:
    dataDistribution: A RddDataDistribution attribute.
    diskUsed: A string attribute.
    memoryUsed: A string attribute.
    name: A string attribute.
    numCachedPartitions: A integer attribute.
    numPartitions: A integer attribute.
    partitions: A RddPartitionInfo attribute.
    rddStorageId: A integer attribute.
    storageLevel: A string attribute.
  """
    dataDistribution = _messages.MessageField('RddDataDistribution', 1, repeated=True)
    diskUsed = _messages.IntegerField(2)
    memoryUsed = _messages.IntegerField(3)
    name = _messages.StringField(4)
    numCachedPartitions = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    numPartitions = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    partitions = _messages.MessageField('RddPartitionInfo', 7, repeated=True)
    rddStorageId = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    storageLevel = _messages.StringField(9)