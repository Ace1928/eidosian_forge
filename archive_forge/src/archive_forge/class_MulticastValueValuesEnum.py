from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MulticastValueValuesEnum(_messages.Enum):
    """Specifies which type of multicast is supported.

    Values:
      MULTICAST_SDN: <no description>
      MULTICAST_ULL: <no description>
      MULTICAST_UNSPECIFIED: <no description>
    """
    MULTICAST_SDN = 0
    MULTICAST_ULL = 1
    MULTICAST_UNSPECIFIED = 2