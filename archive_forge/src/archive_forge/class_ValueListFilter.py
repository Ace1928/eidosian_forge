from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValueListFilter(_messages.Message):
    """A list of values to filter by in ConditionalColumnSetValue

  Enums:
    ValuePresentListValueValuesEnum: Required. Indicates whether the filter
      matches rows with values that are present in the list or those with
      values not present in it.

  Fields:
    ignoreCase: Required. Whether to ignore case when filtering by values.
      Defaults to false
    valuePresentList: Required. Indicates whether the filter matches rows with
      values that are present in the list or those with values not present in
      it.
    values: Required. The list to be used to filter by
  """

    class ValuePresentListValueValuesEnum(_messages.Enum):
        """Required. Indicates whether the filter matches rows with values that
    are present in the list or those with values not present in it.

    Values:
      VALUE_PRESENT_IN_LIST_UNSPECIFIED: Value present in list unspecified
      VALUE_PRESENT_IN_LIST_IF_VALUE_LIST: If the source value is in the
        supplied list at value_list
      VALUE_PRESENT_IN_LIST_IF_VALUE_NOT_LIST: If the source value is not in
        the supplied list at value_list
    """
        VALUE_PRESENT_IN_LIST_UNSPECIFIED = 0
        VALUE_PRESENT_IN_LIST_IF_VALUE_LIST = 1
        VALUE_PRESENT_IN_LIST_IF_VALUE_NOT_LIST = 2
    ignoreCase = _messages.BooleanField(1)
    valuePresentList = _messages.EnumField('ValuePresentListValueValuesEnum', 2)
    values = _messages.StringField(3, repeated=True)