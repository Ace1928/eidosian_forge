from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipRoleRestrictionEvaluation(_messages.Message):
    """The evaluated state of this restriction.

  Enums:
    StateValueValuesEnum: Output only. The current state of the restriction

  Fields:
    state: Output only. The current state of the restriction
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the restriction

    Values:
      STATE_UNSPECIFIED: Default. Should not be used.
      COMPLIANT: The member adheres to the parent group's restriction.
      FORWARD_COMPLIANT: The group-group membership might be currently
        violating some parent group's restriction but in future, it will never
        allow any new member in the child group which can violate parent
        group's restriction.
      NON_COMPLIANT: The member violates the parent group's restriction.
      EVALUATING: The state of the membership is under evaluation.
    """
        STATE_UNSPECIFIED = 0
        COMPLIANT = 1
        FORWARD_COMPLIANT = 2
        NON_COMPLIANT = 3
        EVALUATING = 4
    state = _messages.EnumField('StateValueValuesEnum', 1)