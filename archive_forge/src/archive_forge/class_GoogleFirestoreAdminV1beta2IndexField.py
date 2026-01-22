from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta2IndexField(_messages.Message):
    """A field in an index. The field_path describes which field is indexed,
  the value_mode describes how the field value is indexed.

  Enums:
    ArrayConfigValueValuesEnum: Indicates that this field supports operations
      on `array_value`s.
    OrderValueValuesEnum: Indicates that this field supports ordering by the
      specified order or comparing using =, <, <=, >, >=.

  Fields:
    arrayConfig: Indicates that this field supports operations on
      `array_value`s.
    fieldPath: Can be __name__. For single field indexes, this must match the
      name of the field or may be omitted.
    order: Indicates that this field supports ordering by the specified order
      or comparing using =, <, <=, >, >=.
  """

    class ArrayConfigValueValuesEnum(_messages.Enum):
        """Indicates that this field supports operations on `array_value`s.

    Values:
      ARRAY_CONFIG_UNSPECIFIED: The index does not support additional array
        queries.
      CONTAINS: The index supports array containment queries.
    """
        ARRAY_CONFIG_UNSPECIFIED = 0
        CONTAINS = 1

    class OrderValueValuesEnum(_messages.Enum):
        """Indicates that this field supports ordering by the specified order or
    comparing using =, <, <=, >, >=.

    Values:
      ORDER_UNSPECIFIED: The ordering is unspecified. Not a valid option.
      ASCENDING: The field is ordered by ascending field value.
      DESCENDING: The field is ordered by descending field value.
    """
        ORDER_UNSPECIFIED = 0
        ASCENDING = 1
        DESCENDING = 2
    arrayConfig = _messages.EnumField('ArrayConfigValueValuesEnum', 1)
    fieldPath = _messages.StringField(2)
    order = _messages.EnumField('OrderValueValuesEnum', 3)