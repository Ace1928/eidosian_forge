from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DataCollector(_messages.Message):
    """Data collector configuration.

  Enums:
    TypeValueValuesEnum: Immutable. The type of data this data collector will
      collect.

  Fields:
    createdAt: Output only. The time at which the data collector was created
      in milliseconds since the epoch.
    description: A description of the data collector.
    lastModifiedAt: Output only. The time at which the Data Collector was last
      updated in milliseconds since the epoch.
    name: ID of the data collector. Must begin with `dc_`.
    type: Immutable. The type of data this data collector will collect.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Immutable. The type of data this data collector will collect.

    Values:
      TYPE_UNSPECIFIED: For future compatibility.
      INTEGER: For integer values.
      FLOAT: For float values.
      STRING: For string values.
      BOOLEAN: For boolean values.
      DATETIME: For datetime values.
    """
        TYPE_UNSPECIFIED = 0
        INTEGER = 1
        FLOAT = 2
        STRING = 3
        BOOLEAN = 4
        DATETIME = 5
    createdAt = _messages.IntegerField(1)
    description = _messages.StringField(2)
    lastModifiedAt = _messages.IntegerField(3)
    name = _messages.StringField(4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)