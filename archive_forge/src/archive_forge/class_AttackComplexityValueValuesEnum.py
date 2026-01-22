from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttackComplexityValueValuesEnum(_messages.Enum):
    """This metric describes the conditions beyond the attacker's control
    that must exist in order to exploit the vulnerability.

    Values:
      ATTACK_COMPLEXITY_UNSPECIFIED: Invalid value.
      ATTACK_COMPLEXITY_LOW: Specialized access conditions or extenuating
        circumstances do not exist. An attacker can expect repeatable success
        when attacking the vulnerable component.
      ATTACK_COMPLEXITY_HIGH: A successful attack depends on conditions beyond
        the attacker's control. That is, a successful attack cannot be
        accomplished at will, but requires the attacker to invest in some
        measurable amount of effort in preparation or execution against the
        vulnerable component before a successful attack can be expected.
    """
    ATTACK_COMPLEXITY_UNSPECIFIED = 0
    ATTACK_COMPLEXITY_LOW = 1
    ATTACK_COMPLEXITY_HIGH = 2