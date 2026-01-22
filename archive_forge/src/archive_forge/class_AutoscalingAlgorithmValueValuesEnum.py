from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingAlgorithmValueValuesEnum(_messages.Enum):
    """The algorithm to use for autoscaling

    Values:
      AUTOSCALING_ALGORITHM_UNKNOWN: The algorithm is unknown, or unspecified.
      AUTOSCALING_ALGORITHM_NONE: Disable autoscaling.
      AUTOSCALING_ALGORITHM_BASIC: Increase worker count over time to reduce
        job execution time.
    """
    AUTOSCALING_ALGORITHM_UNKNOWN = 0
    AUTOSCALING_ALGORITHM_NONE = 1
    AUTOSCALING_ALGORITHM_BASIC = 2