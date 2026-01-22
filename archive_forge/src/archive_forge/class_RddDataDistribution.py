from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RddDataDistribution(_messages.Message):
    """Details about RDD usage.

  Fields:
    address: A string attribute.
    diskUsed: A string attribute.
    memoryRemaining: A string attribute.
    memoryUsed: A string attribute.
    offHeapMemoryRemaining: A string attribute.
    offHeapMemoryUsed: A string attribute.
    onHeapMemoryRemaining: A string attribute.
    onHeapMemoryUsed: A string attribute.
  """
    address = _messages.StringField(1)
    diskUsed = _messages.IntegerField(2)
    memoryRemaining = _messages.IntegerField(3)
    memoryUsed = _messages.IntegerField(4)
    offHeapMemoryRemaining = _messages.IntegerField(5)
    offHeapMemoryUsed = _messages.IntegerField(6)
    onHeapMemoryRemaining = _messages.IntegerField(7)
    onHeapMemoryUsed = _messages.IntegerField(8)