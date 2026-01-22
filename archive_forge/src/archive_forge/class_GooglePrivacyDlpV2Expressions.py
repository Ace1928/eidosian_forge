from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Expressions(_messages.Message):
    """An expression, consisting of an operator and conditions.

  Enums:
    LogicalOperatorValueValuesEnum: The operator to apply to the result of
      conditions. Default and currently only supported value is `AND`.

  Fields:
    conditions: Conditions to apply to the expression.
    logicalOperator: The operator to apply to the result of conditions.
      Default and currently only supported value is `AND`.
  """

    class LogicalOperatorValueValuesEnum(_messages.Enum):
        """The operator to apply to the result of conditions. Default and
    currently only supported value is `AND`.

    Values:
      LOGICAL_OPERATOR_UNSPECIFIED: Unused
      AND: Conditional AND
    """
        LOGICAL_OPERATOR_UNSPECIFIED = 0
        AND = 1
    conditions = _messages.MessageField('GooglePrivacyDlpV2Conditions', 1)
    logicalOperator = _messages.EnumField('LogicalOperatorValueValuesEnum', 2)