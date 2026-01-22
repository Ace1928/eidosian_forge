from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1Constraint(_messages.Message):
    """The definition of a constraint.

  Enums:
    ConstraintDefaultValueValuesEnum: The evaluation behavior of this
      constraint in the absence of 'Policy'.

  Fields:
    booleanConstraint: Defines this constraint as being a BooleanConstraint.
    constraintDefault: The evaluation behavior of this constraint in the
      absence of 'Policy'.
    description: Detailed description of what this `Constraint` controls as
      well as how and where it is enforced.
    displayName: The human readable name of the constraint.
    listConstraint: Defines this constraint as being a ListConstraint.
    name: The unique name of the constraint. Format of the name should be *
      `constraints/{constraint_name}` For example,
      `constraints/compute.disableSerialPortAccess`.
  """

    class ConstraintDefaultValueValuesEnum(_messages.Enum):
        """The evaluation behavior of this constraint in the absence of 'Policy'.

    Values:
      CONSTRAINT_DEFAULT_UNSPECIFIED: This is only used for distinguishing
        unset values and should never be used.
      ALLOW: Indicate that all values are allowed for list constraints.
        Indicate that enforcement is off for boolean constraints.
      DENY: Indicate that all values are denied for list constraints. Indicate
        that enforcement is on for boolean constraints.
    """
        CONSTRAINT_DEFAULT_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2
    booleanConstraint = _messages.MessageField('GoogleCloudAssetV1BooleanConstraint', 1)
    constraintDefault = _messages.EnumField('ConstraintDefaultValueValuesEnum', 2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    listConstraint = _messages.MessageField('GoogleCloudAssetV1ListConstraint', 5)
    name = _messages.StringField(6)