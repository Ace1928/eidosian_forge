from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogicalExpression(_messages.Message):
    """Struct for representing boolean expressions.

  Enums:
    LogicalOperatorValueValuesEnum: The logical operator to use between the
      fields and conditions.

  Fields:
    fieldComparisons: A list of fields to be compared.
    logicalExpressions: A list of nested conditions to be compared.
    logicalOperator: The logical operator to use between the fields and
      conditions.
  """

    class LogicalOperatorValueValuesEnum(_messages.Enum):
        """The logical operator to use between the fields and conditions.

    Values:
      OPERATOR_UNSPECIFIED: The default value.
      AND: AND operator; The conditions must all be true.
      OR: OR operator; At least one of the conditions must be true.
    """
        OPERATOR_UNSPECIFIED = 0
        AND = 1
        OR = 2
    fieldComparisons = _messages.MessageField('FieldComparison', 1, repeated=True)
    logicalExpressions = _messages.MessageField('LogicalExpression', 2, repeated=True)
    logicalOperator = _messages.EnumField('LogicalOperatorValueValuesEnum', 3)