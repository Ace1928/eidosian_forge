from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UtilizationTargetTypeValueValuesEnum(_messages.Enum):
    """Defines how target utilization value is expressed for a Stackdriver
    Monitoring metric. Either GAUGE, DELTA_PER_SECOND, or DELTA_PER_MINUTE.

    Values:
      DELTA_PER_MINUTE: Sets the utilization target value for a cumulative or
        delta metric, expressed as the rate of growth per minute.
      DELTA_PER_SECOND: Sets the utilization target value for a cumulative or
        delta metric, expressed as the rate of growth per second.
      GAUGE: Sets the utilization target value for a gauge metric. The
        autoscaler will collect the average utilization of the virtual
        machines from the last couple of minutes, and compare the value to the
        utilization target value to perform autoscaling.
    """
    DELTA_PER_MINUTE = 0
    DELTA_PER_SECOND = 1
    GAUGE = 2