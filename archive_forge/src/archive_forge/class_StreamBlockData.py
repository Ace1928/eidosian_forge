from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamBlockData(_messages.Message):
    """Stream Block Data.

  Fields:
    deserialized: A boolean attribute.
    diskSize: A string attribute.
    executorId: A string attribute.
    hostPort: A string attribute.
    memSize: A string attribute.
    name: A string attribute.
    storageLevel: A string attribute.
    useDisk: A boolean attribute.
    useMemory: A boolean attribute.
  """
    deserialized = _messages.BooleanField(1)
    diskSize = _messages.IntegerField(2)
    executorId = _messages.StringField(3)
    hostPort = _messages.StringField(4)
    memSize = _messages.IntegerField(5)
    name = _messages.StringField(6)
    storageLevel = _messages.StringField(7)
    useDisk = _messages.BooleanField(8)
    useMemory = _messages.BooleanField(9)