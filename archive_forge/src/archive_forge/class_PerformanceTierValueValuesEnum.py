from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerformanceTierValueValuesEnum(_messages.Enum):
    """Performance tier of the Volume. Default is SHARED.

    Values:
      VOLUME_PERFORMANCE_TIER_UNSPECIFIED: Value is not specified.
      VOLUME_PERFORMANCE_TIER_SHARED: Regular volumes, shared aggregates.
      VOLUME_PERFORMANCE_TIER_ASSIGNED: Assigned aggregates.
      VOLUME_PERFORMANCE_TIER_HT: High throughput aggregates.
      VOLUME_PERFORMANCE_TIER_QOS2_PERFORMANCE: QoS 2.0 high performance
        storage.
    """
    VOLUME_PERFORMANCE_TIER_UNSPECIFIED = 0
    VOLUME_PERFORMANCE_TIER_SHARED = 1
    VOLUME_PERFORMANCE_TIER_ASSIGNED = 2
    VOLUME_PERFORMANCE_TIER_HT = 3
    VOLUME_PERFORMANCE_TIER_QOS2_PERFORMANCE = 4