from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImpactValueValuesEnum(_messages.Enum):
    """The potential impact of the vulnerability if it was to be exploited.

    Values:
      RISK_RATING_UNSPECIFIED: Invalid or empty value.
      LOW: Exploitation would have little to no security impact.
      MEDIUM: Exploitation would enable attackers to perform activities, or
        could allow attackers to have a direct impact, but would require
        additional steps.
      HIGH: Exploitation would enable attackers to have a notable direct
        impact without needing to overcome any major mitigating factors.
      CRITICAL: Exploitation would fundamentally undermine the security of
        affected systems, enable actors to perform significant attacks with
        minimal effort, with little to no mitigating factors to overcome.
    """
    RISK_RATING_UNSPECIFIED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4