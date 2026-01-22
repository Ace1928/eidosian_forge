from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FieldFilter(_messages.Message):
    """A filter on a specific field.

  Enums:
    OpValueValuesEnum: The operator to filter by.

  Fields:
    field: The field to filter by.
    op: The operator to filter by.
    value: The value to compare to.
  """

    class OpValueValuesEnum(_messages.Enum):
        """The operator to filter by.

    Values:
      OPERATOR_UNSPECIFIED: Unspecified. This value must not be used.
      LESS_THAN: The given `field` is less than the given `value`. Requires: *
        That `field` come first in `order_by`.
      LESS_THAN_OR_EQUAL: The given `field` is less than or equal to the given
        `value`. Requires: * That `field` come first in `order_by`.
      GREATER_THAN: The given `field` is greater than the given `value`.
        Requires: * That `field` come first in `order_by`.
      GREATER_THAN_OR_EQUAL: The given `field` is greater than or equal to the
        given `value`. Requires: * That `field` come first in `order_by`.
      EQUAL: The given `field` is equal to the given `value`.
      NOT_EQUAL: The given `field` is not equal to the given `value`.
        Requires: * No other `NOT_EQUAL`, `NOT_IN`, `IS_NOT_NULL`, or
        `IS_NOT_NAN`. * That `field` comes first in the `order_by`.
      ARRAY_CONTAINS: The given `field` is an array that contains the given
        `value`.
      IN: The given `field` is equal to at least one value in the given array.
        Requires: * That `value` is a non-empty `ArrayValue`, subject to
        disjunction limits. * No `NOT_IN` filters in the same query.
      ARRAY_CONTAINS_ANY: The given `field` is an array that contains any of
        the values in the given array. Requires: * That `value` is a non-empty
        `ArrayValue`, subject to disjunction limits. * No other
        `ARRAY_CONTAINS_ANY` filters within the same disjunction. * No
        `NOT_IN` filters in the same query.
      NOT_IN: The value of the `field` is not in the given array. Requires: *
        That `value` is a non-empty `ArrayValue` with at most 10 values. * No
        other `OR`, `IN`, `ARRAY_CONTAINS_ANY`, `NOT_IN`, `NOT_EQUAL`,
        `IS_NOT_NULL`, or `IS_NOT_NAN`. * That `field` comes first in the
        `order_by`.
    """
        OPERATOR_UNSPECIFIED = 0
        LESS_THAN = 1
        LESS_THAN_OR_EQUAL = 2
        GREATER_THAN = 3
        GREATER_THAN_OR_EQUAL = 4
        EQUAL = 5
        NOT_EQUAL = 6
        ARRAY_CONTAINS = 7
        IN = 8
        ARRAY_CONTAINS_ANY = 9
        NOT_IN = 10
    field = _messages.MessageField('FieldReference', 1)
    op = _messages.EnumField('OpValueValuesEnum', 2)
    value = _messages.MessageField('Value', 3)