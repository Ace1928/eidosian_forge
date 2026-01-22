from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DriverSchedulingConfig(_messages.Message):
    """Driver scheduling configuration.

  Fields:
    memoryMb: Required. The amount of memory in MB the driver is requesting.
    vcores: Required. The number of vCPUs the driver is requesting.
  """
    memoryMb = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    vcores = _messages.IntegerField(2, variant=_messages.Variant.INT32)