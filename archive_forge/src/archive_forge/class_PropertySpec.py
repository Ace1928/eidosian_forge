from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PropertySpec(_messages.Message):
    """PropertySpec holds information about a property in an object.

  Enums:
    TypeValueValuesEnum: A type for the object.

  Fields:
    type: A type for the object.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """A type for the object.

    Values:
      TYPE_UNSPECIFIED: Default enum type; should not be used.
      STRING: Default
    """
        TYPE_UNSPECIFIED = 0
        STRING = 1
    type = _messages.EnumField('TypeValueValuesEnum', 1)