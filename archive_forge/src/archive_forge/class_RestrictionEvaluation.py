from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestrictionEvaluation(_messages.Message):
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
      EVALUATING: The restriction state is currently being evaluated.
      COMPLIANT: All transitive memberships are adhering to restriction.
      FORWARD_COMPLIANT: Some transitive memberships violate the restriction.
        No new violating memberships can be added.
      NON_COMPLIANT: Some transitive memberships violate the restriction. New
        violating direct memberships will be denied while indirect memberships
        may be added.
    """
        STATE_UNSPECIFIED = 0
        EVALUATING = 1
        COMPLIANT = 2
        FORWARD_COMPLIANT = 3
        NON_COMPLIANT = 4
    state = _messages.EnumField('StateValueValuesEnum', 1)