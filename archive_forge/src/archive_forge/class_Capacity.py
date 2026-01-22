from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Capacity(_messages.Message):
    """The throughput capacity configuration for each partition.

  Fields:
    publishMibPerSec: Publish throughput capacity per partition in MiB/s. Must
      be >= 4 and <= 16.
    subscribeMibPerSec: Subscribe throughput capacity per partition in MiB/s.
      Must be >= 4 and <= 32.
  """
    publishMibPerSec = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    subscribeMibPerSec = _messages.IntegerField(2, variant=_messages.Variant.INT32)