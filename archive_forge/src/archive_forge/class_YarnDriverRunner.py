from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class YarnDriverRunner(_messages.Message):
    """Schedule the driver on worker nodes using YARN.

  Fields:
    memoryMb: Optional. The amount of memory in MB the driver is requesting
      from YARN.
    vcores: Optional. The number of vCPUs this driver is requesting from YARN.
  """
    memoryMb = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    vcores = _messages.IntegerField(2, variant=_messages.Variant.INT32)