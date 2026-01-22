from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1SettingMetadata(_messages.Message):
    """Metadata about a setting which is not editable by the end user.

  Enums:
    DataTypeValueValuesEnum: The data type for this setting.

  Fields:
    dataType: The data type for this setting.
    defaultValue: The value provided by Setting.effective_value if no setting
      value is explicitly set. Note: not all settings have a default value.
    description: A detailed description of what this setting does.
    displayName: The human readable name for this setting.
    readOnly: A flag indicating that values of this setting cannot be
      modified. See documentation for the specific setting for updates and
      reasons.
  """

    class DataTypeValueValuesEnum(_messages.Enum):
        """The data type for this setting.

    Values:
      DATA_TYPE_UNSPECIFIED: Unspecified data type.
      BOOLEAN: A boolean setting.
      STRING: A string setting.
      STRING_SET: A string set setting.
      ENUM_VALUE: A Enum setting
      DURATION_VALUE: A Duration setting
      STRING_MAP: A string->string map setting
    """
        DATA_TYPE_UNSPECIFIED = 0
        BOOLEAN = 1
        STRING = 2
        STRING_SET = 3
        ENUM_VALUE = 4
        DURATION_VALUE = 5
        STRING_MAP = 6
    dataType = _messages.EnumField('DataTypeValueValuesEnum', 1)
    defaultValue = _messages.MessageField('GoogleCloudResourcesettingsV1Value', 2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    readOnly = _messages.BooleanField(5)