from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PubSubExpressions(_messages.Message):
    """An expression, consisting of an operator and conditions.

  Enums:
    LogicalOperatorValueValuesEnum: The operator to apply to the collection of
      conditions.

  Fields:
    conditions: Conditions to apply to the expression.
    logicalOperator: The operator to apply to the collection of conditions.
  """

    class LogicalOperatorValueValuesEnum(_messages.Enum):
        """The operator to apply to the collection of conditions.

    Values:
      LOGICAL_OPERATOR_UNSPECIFIED: Unused.
      OR: Conditional OR.
      AND: Conditional AND.
    """
        LOGICAL_OPERATOR_UNSPECIFIED = 0
        OR = 1
        AND = 2
    conditions = _messages.MessageField('GooglePrivacyDlpV2PubSubCondition', 1, repeated=True)
    logicalOperator = _messages.EnumField('LogicalOperatorValueValuesEnum', 2)