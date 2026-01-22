from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DriverLogLevelsValue(_messages.Message):
    """The per-package log levels for the driver. This can include "root"
    package name to configure rootLogger. Examples: - 'com.google = FATAL' -
    'root = INFO' - 'org.apache = DEBUG'

    Messages:
      AdditionalProperty: An additional property for a DriverLogLevelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DriverLogLevelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DriverLogLevelsValue object.

      Enums:
        ValueValueValuesEnum:

      Fields:
        key: Name of the additional property.
        value: A ValueValueValuesEnum attribute.
      """

        class ValueValueValuesEnum(_messages.Enum):
            """ValueValueValuesEnum enum type.

        Values:
          LEVEL_UNSPECIFIED: Level is unspecified. Use default level for
            log4j.
          ALL: Use ALL level for log4j.
          TRACE: Use TRACE level for log4j.
          DEBUG: Use DEBUG level for log4j.
          INFO: Use INFO level for log4j.
          WARN: Use WARN level for log4j.
          ERROR: Use ERROR level for log4j.
          FATAL: Use FATAL level for log4j.
          OFF: Turn off log4j.
        """
            LEVEL_UNSPECIFIED = 0
            ALL = 1
            TRACE = 2
            DEBUG = 3
            INFO = 4
            WARN = 5
            ERROR = 6
            FATAL = 7
            OFF = 8
        key = _messages.StringField(1)
        value = _messages.EnumField('ValueValueValuesEnum', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)