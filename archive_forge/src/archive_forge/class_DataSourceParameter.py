from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataSourceParameter(_messages.Message):
    """A parameter used to define custom fields in a data source definition.

  Enums:
    TypeValueValuesEnum: Parameter type.

  Fields:
    allowedValues: All possible values for the parameter.
    deprecated: If true, it should not be used in new transfers, and it should
      not be visible to users.
    description: Parameter description.
    displayName: Parameter display name in the user interface.
    fields: Deprecated. This field has no effect.
    immutable: Cannot be changed after initial creation.
    maxValue: For integer and double values specifies maximum allowed value.
    minValue: For integer and double values specifies minimum allowed value.
    paramId: Parameter identifier.
    recurse: Deprecated. This field has no effect.
    repeated: Deprecated. This field has no effect.
    required: Is parameter required.
    type: Parameter type.
    validationDescription: Description of the requirements for this field, in
      case the user input does not fulfill the regex pattern or min/max
      values.
    validationHelpUrl: URL to a help document to further explain the naming
      requirements.
    validationRegex: Regular expression which can be used for parameter
      validation.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Parameter type.

    Values:
      TYPE_UNSPECIFIED: Type unspecified.
      STRING: String parameter.
      INTEGER: Integer parameter (64-bits). Will be serialized to json as
        string.
      DOUBLE: Double precision floating point parameter.
      BOOLEAN: Boolean parameter.
      RECORD: Deprecated. This field has no effect.
      PLUS_PAGE: Page ID for a Google+ Page.
      LIST: List of strings parameter.
    """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        INTEGER = 2
        DOUBLE = 3
        BOOLEAN = 4
        RECORD = 5
        PLUS_PAGE = 6
        LIST = 7
    allowedValues = _messages.StringField(1, repeated=True)
    deprecated = _messages.BooleanField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    fields = _messages.MessageField('DataSourceParameter', 5, repeated=True)
    immutable = _messages.BooleanField(6)
    maxValue = _messages.FloatField(7)
    minValue = _messages.FloatField(8)
    paramId = _messages.StringField(9)
    recurse = _messages.BooleanField(10)
    repeated = _messages.BooleanField(11)
    required = _messages.BooleanField(12)
    type = _messages.EnumField('TypeValueValuesEnum', 13)
    validationDescription = _messages.StringField(14)
    validationHelpUrl = _messages.StringField(15)
    validationRegex = _messages.StringField(16)