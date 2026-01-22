from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PointInTimeRecoveryEnablementValueValuesEnum(_messages.Enum):
    """Whether to enable the PITR feature on this database.

    Values:
      POINT_IN_TIME_RECOVERY_ENABLEMENT_UNSPECIFIED: Not used.
      POINT_IN_TIME_RECOVERY_ENABLED: Reads are supported on selected versions
        of the data from within the past 7 days: * Reads against any timestamp
        within the past hour * Reads against 1-minute snapshots beyond 1 hour
        and within 7 days `version_retention_period` and
        `earliest_version_time` can be used to determine the supported
        versions.
      POINT_IN_TIME_RECOVERY_DISABLED: Reads are supported on any version of
        the data from within the past 1 hour.
    """
    POINT_IN_TIME_RECOVERY_ENABLEMENT_UNSPECIFIED = 0
    POINT_IN_TIME_RECOVERY_ENABLED = 1
    POINT_IN_TIME_RECOVERY_DISABLED = 2