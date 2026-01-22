from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationConfigTemplate(_messages.Message):
    """DestinationConfigTemplate defines required destinations supported by the
  Connector.

  Enums:
    PortFieldTypeValueValuesEnum: Whether port number should be provided by
      customers.

  Fields:
    defaultPort: The default port.
    description: Description.
    displayName: Display name of the parameter.
    isAdvanced: Whether the current destination tempalate is part of Advanced
      settings
    key: Key of the destination.
    max: The maximum number of destinations supported for this key.
    min: The minimum number of destinations supported for this key.
    portFieldType: Whether port number should be provided by customers.
    regexPattern: Regex pattern for host.
  """

    class PortFieldTypeValueValuesEnum(_messages.Enum):
        """Whether port number should be provided by customers.

    Values:
      FIELD_TYPE_UNSPECIFIED: <no description>
      REQUIRED: <no description>
      OPTIONAL: <no description>
      NOT_USED: <no description>
    """
        FIELD_TYPE_UNSPECIFIED = 0
        REQUIRED = 1
        OPTIONAL = 2
        NOT_USED = 3
    defaultPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    isAdvanced = _messages.BooleanField(4)
    key = _messages.StringField(5)
    max = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    min = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    portFieldType = _messages.EnumField('PortFieldTypeValueValuesEnum', 8)
    regexPattern = _messages.StringField(9)