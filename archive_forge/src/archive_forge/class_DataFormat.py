from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataFormat(_messages.Message):
    """The data format of a message payload.

  Enums:
    TypeValueValuesEnum: Required. The format type of a message payload.

  Fields:
    schema: Optional. The schema of a message payload.
    type: Required. The format type of a message payload.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The format type of a message payload.

    Values:
      TYPE_UNSPECIFIED: FORMAT types unspecified.
      JSON: JSON
      PROTOCOL_BUFFERS: PROTO
      AVRO: AVRO
    """
        TYPE_UNSPECIFIED = 0
        JSON = 1
        PROTOCOL_BUFFERS = 2
        AVRO = 3
    schema = _messages.MessageField('Schema', 1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)