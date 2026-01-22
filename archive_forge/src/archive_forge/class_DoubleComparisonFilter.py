from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DoubleComparisonFilter(_messages.Message):
    """Filter based on relation between source value and compare value of type
  double in ConditionalColumnSetValue

  Enums:
    ValueComparisonValueValuesEnum: Required. Relation between source value
      and compare value

  Fields:
    value: Required. Double compare value to be used
    valueComparison: Required. Relation between source value and compare value
  """

    class ValueComparisonValueValuesEnum(_messages.Enum):
        """Required. Relation between source value and compare value

    Values:
      VALUE_COMPARISON_UNSPECIFIED: Value comparison unspecified.
      VALUE_COMPARISON_IF_VALUE_SMALLER_THAN: Value is smaller than the
        Compare value.
      VALUE_COMPARISON_IF_VALUE_SMALLER_EQUAL_THAN: Value is smaller or equal
        than the Compare value.
      VALUE_COMPARISON_IF_VALUE_LARGER_THAN: Value is larger than the Compare
        value.
      VALUE_COMPARISON_IF_VALUE_LARGER_EQUAL_THAN: Value is larger or equal
        than the Compare value.
    """
        VALUE_COMPARISON_UNSPECIFIED = 0
        VALUE_COMPARISON_IF_VALUE_SMALLER_THAN = 1
        VALUE_COMPARISON_IF_VALUE_SMALLER_EQUAL_THAN = 2
        VALUE_COMPARISON_IF_VALUE_LARGER_THAN = 3
        VALUE_COMPARISON_IF_VALUE_LARGER_EQUAL_THAN = 4
    value = _messages.FloatField(1)
    valueComparison = _messages.EnumField('ValueComparisonValueValuesEnum', 2)