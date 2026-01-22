from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PropertyOrder(_messages.Message):
    """The desired order for a specific property.

  Enums:
    DirectionValueValuesEnum: The direction to order by. Defaults to
      `ASCENDING`.

  Fields:
    direction: The direction to order by. Defaults to `ASCENDING`.
    property: The property to order by.
  """

    class DirectionValueValuesEnum(_messages.Enum):
        """The direction to order by. Defaults to `ASCENDING`.

    Values:
      DIRECTION_UNSPECIFIED: Unspecified. This value must not be used.
      ASCENDING: Ascending.
      DESCENDING: Descending.
    """
        DIRECTION_UNSPECIFIED = 0
        ASCENDING = 1
        DESCENDING = 2
    direction = _messages.EnumField('DirectionValueValuesEnum', 1)
    property = _messages.MessageField('PropertyReference', 2)