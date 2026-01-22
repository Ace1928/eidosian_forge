from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Condition(_messages.Message):
    """The field type of `value` and `field` do not need to match to be
  considered equal, but not all comparisons are possible. EQUAL_TO and
  NOT_EQUAL_TO attempt to compare even with incompatible types, but all other
  comparisons are invalid with incompatible types. A `value` of type: -
  `string` can be compared against all other types - `boolean` can only be
  compared against other booleans - `integer` can be compared against doubles
  or a string if the string value can be parsed as an integer. - `double` can
  be compared against integers or a string if the string can be parsed as a
  double. - `Timestamp` can be compared against strings in RFC 3339 date
  string format. - `TimeOfDay` can be compared against timestamps and strings
  in the format of 'HH:mm:ss'. If we fail to compare do to type mismatch, a
  warning will be given and the condition will evaluate to false.

  Enums:
    OperatorValueValuesEnum: Required. Operator used to compare the field or
      infoType to the value.

  Fields:
    field: Required. Field within the record this condition is evaluated
      against.
    operator: Required. Operator used to compare the field or infoType to the
      value.
    value: Value to compare against. [Mandatory, except for `EXISTS` tests.]
  """

    class OperatorValueValuesEnum(_messages.Enum):
        """Required. Operator used to compare the field or infoType to the value.

    Values:
      RELATIONAL_OPERATOR_UNSPECIFIED: Unused
      EQUAL_TO: Equal. Attempts to match even with incompatible types.
      NOT_EQUAL_TO: Not equal to. Attempts to match even with incompatible
        types.
      GREATER_THAN: Greater than.
      LESS_THAN: Less than.
      GREATER_THAN_OR_EQUALS: Greater than or equals.
      LESS_THAN_OR_EQUALS: Less than or equals.
      EXISTS: Exists
    """
        RELATIONAL_OPERATOR_UNSPECIFIED = 0
        EQUAL_TO = 1
        NOT_EQUAL_TO = 2
        GREATER_THAN = 3
        LESS_THAN = 4
        GREATER_THAN_OR_EQUALS = 5
        LESS_THAN_OR_EQUALS = 6
        EXISTS = 7
    field = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1)
    operator = _messages.EnumField('OperatorValueValuesEnum', 2)
    value = _messages.MessageField('GooglePrivacyDlpV2Value', 3)