from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VirtualCpuTypeValueValuesEnum(_messages.Enum):
    """Required. The processor type of this instance.

    Values:
      VIRTUAL_CPU_TYPE_UNSPECIFIED: Unspecified.
      DEDICATED: Dedicated processors. Processor counts for this type must be
        whole numbers.
      UNCAPPED_SHARED: Uncapped shared processors.
      CAPPED_SHARED: Capped shared processors.
    """
    VIRTUAL_CPU_TYPE_UNSPECIFIED = 0
    DEDICATED = 1
    UNCAPPED_SHARED = 2
    CAPPED_SHARED = 3