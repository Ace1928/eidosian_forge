from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemoryMetrics(_messages.Message):
    """A MemoryMetrics object.

  Fields:
    totalOffHeapStorageMemory: A string attribute.
    totalOnHeapStorageMemory: A string attribute.
    usedOffHeapStorageMemory: A string attribute.
    usedOnHeapStorageMemory: A string attribute.
  """
    totalOffHeapStorageMemory = _messages.IntegerField(1)
    totalOnHeapStorageMemory = _messages.IntegerField(2)
    usedOffHeapStorageMemory = _messages.IntegerField(3)
    usedOnHeapStorageMemory = _messages.IntegerField(4)