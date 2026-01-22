from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceNumericFilter(_messages.Message):
    """Filter for fixed point number data types such as NUMERIC/NUMBER

  Enums:
    NumericFilterOptionValueValuesEnum: Required. Enum to set the option
      defining the datatypes numeric filter has to be applied to

  Fields:
    numericFilterOption: Required. Enum to set the option defining the
      datatypes numeric filter has to be applied to
    sourceMaxPrecisionFilter: Optional. The filter will match columns with
      precision smaller than or equal to this number.
    sourceMaxScaleFilter: Optional. The filter will match columns with scale
      smaller than or equal to this number.
    sourceMinPrecisionFilter: Optional. The filter will match columns with
      precision greater than or equal to this number.
    sourceMinScaleFilter: Optional. The filter will match columns with scale
      greater than or equal to this number.
  """

    class NumericFilterOptionValueValuesEnum(_messages.Enum):
        """Required. Enum to set the option defining the datatypes numeric filter
    has to be applied to

    Values:
      NUMERIC_FILTER_OPTION_UNSPECIFIED: Numeric filter option unspecified
      NUMERIC_FILTER_OPTION_ALL: Numeric filter option that matches all
        numeric columns.
      NUMERIC_FILTER_OPTION_LIMIT: Numeric filter option that matches columns
        having numeric datatypes with specified precision and scale within the
        limited range of filter.
      NUMERIC_FILTER_OPTION_LIMITLESS: Numeric filter option that matches only
        the numeric columns with no precision and scale specified.
    """
        NUMERIC_FILTER_OPTION_UNSPECIFIED = 0
        NUMERIC_FILTER_OPTION_ALL = 1
        NUMERIC_FILTER_OPTION_LIMIT = 2
        NUMERIC_FILTER_OPTION_LIMITLESS = 3
    numericFilterOption = _messages.EnumField('NumericFilterOptionValueValuesEnum', 1)
    sourceMaxPrecisionFilter = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    sourceMaxScaleFilter = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    sourceMinPrecisionFilter = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    sourceMinScaleFilter = _messages.IntegerField(5, variant=_messages.Variant.INT32)