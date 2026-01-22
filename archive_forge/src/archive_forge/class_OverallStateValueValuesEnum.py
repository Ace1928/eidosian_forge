from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverallStateValueValuesEnum(_messages.Enum):
    """Health state of overall health of this instance. Such as
    unhealthy/warning/healthy

    Values:
      HEALTH_STATE_UNSPECIFIED: Invalid
      UNKNOWN: Unknown. May indicate exceptions.
      HEALTHY: Healthy
      WARNING: Warning
      UNHEALTHY: Unhealthy
    """
    HEALTH_STATE_UNSPECIFIED = 0
    UNKNOWN = 1
    HEALTHY = 2
    WARNING = 3
    UNHEALTHY = 4