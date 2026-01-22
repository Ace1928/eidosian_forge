from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DeviceIpBlockCatalog(_messages.Message):
    """List of IP blocks used by the Firebase Test Lab

  Fields:
    ipBlocks: The device IP blocks used by Firebase Test Lab
  """
    ipBlocks = _messages.MessageField('DeviceIpBlock', 1, repeated=True)