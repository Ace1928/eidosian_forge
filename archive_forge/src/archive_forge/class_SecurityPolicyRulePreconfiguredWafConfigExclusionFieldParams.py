from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams(_messages.Message):
    """A SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams object.

  Enums:
    OpValueValuesEnum: The match operator for the field.

  Fields:
    op: The match operator for the field.
    val: The value of the field.
  """

    class OpValueValuesEnum(_messages.Enum):
        """The match operator for the field.

    Values:
      CONTAINS: The operator matches if the field value contains the specified
        value.
      ENDS_WITH: The operator matches if the field value ends with the
        specified value.
      EQUALS: The operator matches if the field value equals the specified
        value.
      EQUALS_ANY: The operator matches if the field value is any value.
      STARTS_WITH: The operator matches if the field value starts with the
        specified value.
    """
        CONTAINS = 0
        ENDS_WITH = 1
        EQUALS = 2
        EQUALS_ANY = 3
        STARTS_WITH = 4
    op = _messages.EnumField('OpValueValuesEnum', 1)
    val = _messages.StringField(2)