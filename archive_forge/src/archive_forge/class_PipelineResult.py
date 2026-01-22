from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineResult(_messages.Message):
    """A value produced by a Pipeline.

  Enums:
    TypeValueValuesEnum: Output only. The type of data that the result holds.

  Fields:
    description: Output only. Description of the result.
    name: Output only. Name of the result.
    type: Output only. The type of data that the result holds.
    value: Output only. Value of the result.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. The type of data that the result holds.

    Values:
      TYPE_UNSPECIFIED: Default enum type; should not be used.
      STRING: Default
      ARRAY: Array type
      OBJECT: Object type
    """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        ARRAY = 2
        OBJECT = 3
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)
    value = _messages.MessageField('ResultValue', 4)