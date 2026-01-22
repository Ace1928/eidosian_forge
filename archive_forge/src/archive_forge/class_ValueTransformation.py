from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValueTransformation(_messages.Message):
    """Description of data transformation during migration as part of the
  ConditionalColumnSetValue.

  Fields:
    applyHash: Optional. Applies a hash function on the data
    assignMaxValue: Optional. Set to max_value - if integer or numeric, will
      use int.maxvalue, etc
    assignMinValue: Optional. Set to min_value - if integer or numeric, will
      use int.minvalue, etc
    assignNull: Optional. Set to null
    assignSpecificValue: Optional. Set to a specific value (value is converted
      to fit the target data type)
    doubleComparison: Optional. Filter on relation between source value and
      compare value of type double.
    intComparison: Optional. Filter on relation between source value and
      compare value of type integer.
    isNull: Optional. Value is null
    roundScale: Optional. Allows the data to change scale
    valueList: Optional. Value is found in the specified list.
  """
    applyHash = _messages.MessageField('ApplyHash', 1)
    assignMaxValue = _messages.MessageField('Empty', 2)
    assignMinValue = _messages.MessageField('Empty', 3)
    assignNull = _messages.MessageField('Empty', 4)
    assignSpecificValue = _messages.MessageField('AssignSpecificValue', 5)
    doubleComparison = _messages.MessageField('DoubleComparisonFilter', 6)
    intComparison = _messages.MessageField('IntComparisonFilter', 7)
    isNull = _messages.MessageField('Empty', 8)
    roundScale = _messages.MessageField('RoundToScale', 9)
    valueList = _messages.MessageField('ValueListFilter', 10)