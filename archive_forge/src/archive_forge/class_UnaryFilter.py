from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UnaryFilter(_messages.Message):
    """A filter with a single operand.

  Enums:
    OpValueValuesEnum: The unary operator to apply.

  Fields:
    field: The field to which to apply the operator.
    op: The unary operator to apply.
  """

    class OpValueValuesEnum(_messages.Enum):
        """The unary operator to apply.

    Values:
      OPERATOR_UNSPECIFIED: Unspecified. This value must not be used.
      IS_NAN: The given `field` is equal to `NaN`.
      IS_NULL: The given `field` is equal to `NULL`.
      IS_NOT_NAN: The given `field` is not equal to `NaN`. Requires: * No
        other `NOT_EQUAL`, `NOT_IN`, `IS_NOT_NULL`, or `IS_NOT_NAN`. * That
        `field` comes first in the `order_by`.
      IS_NOT_NULL: The given `field` is not equal to `NULL`. Requires: * A
        single `NOT_EQUAL`, `NOT_IN`, `IS_NOT_NULL`, or `IS_NOT_NAN`. * That
        `field` comes first in the `order_by`.
    """
        OPERATOR_UNSPECIFIED = 0
        IS_NAN = 1
        IS_NULL = 2
        IS_NOT_NAN = 3
        IS_NOT_NULL = 4
    field = _messages.MessageField('FieldReference', 1)
    op = _messages.EnumField('OpValueValuesEnum', 2)