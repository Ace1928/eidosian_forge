from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RddPartitionInfo(_messages.Message):
    """Information about RDD partitions.

  Fields:
    blockName: A string attribute.
    diskUsed: A string attribute.
    executors: A string attribute.
    memoryUsed: A string attribute.
    storageLevel: A string attribute.
  """
    blockName = _messages.StringField(1)
    diskUsed = _messages.IntegerField(2)
    executors = _messages.StringField(3, repeated=True)
    memoryUsed = _messages.IntegerField(4)
    storageLevel = _messages.StringField(5)