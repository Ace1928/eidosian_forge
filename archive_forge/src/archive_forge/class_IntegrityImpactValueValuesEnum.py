from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntegrityImpactValueValuesEnum(_messages.Enum):
    """This metric measures the impact to integrity of a successfully
    exploited vulnerability.

    Values:
      IMPACT_UNSPECIFIED: Invalid value.
      IMPACT_HIGH: High impact.
      IMPACT_LOW: Low impact.
      IMPACT_NONE: No impact.
    """
    IMPACT_UNSPECIFIED = 0
    IMPACT_HIGH = 1
    IMPACT_LOW = 2
    IMPACT_NONE = 3