from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SyncParallelLevelValueValuesEnum(_messages.Enum):
    """Optional. Parallel level for initial data sync. Currently only
    applicable for PostgreSQL.

    Values:
      EXTERNAL_SYNC_PARALLEL_LEVEL_UNSPECIFIED: Unknown sync parallel level.
        Will be defaulted to OPTIMAL.
      MIN: Minimal parallel level.
      OPTIMAL: Optimal parallel level.
      MAX: Maximum parallel level.
    """
    EXTERNAL_SYNC_PARALLEL_LEVEL_UNSPECIFIED = 0
    MIN = 1
    OPTIMAL = 2
    MAX = 3