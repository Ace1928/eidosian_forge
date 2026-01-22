from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1IndexedProperty(_messages.Message):
    """A property of an index.

  Enums:
    DirectionValueValuesEnum: Required. The indexed property's direction. Must
      not be DIRECTION_UNSPECIFIED.

  Fields:
    direction: Required. The indexed property's direction. Must not be
      DIRECTION_UNSPECIFIED.
    name: Required. The property name to index.
  """

    class DirectionValueValuesEnum(_messages.Enum):
        """Required. The indexed property's direction. Must not be
    DIRECTION_UNSPECIFIED.

    Values:
      DIRECTION_UNSPECIFIED: The direction is unspecified.
      ASCENDING: The property's values are indexed so as to support sequencing
        in ascending order and also query by <, >, <=, >=, and =.
      DESCENDING: The property's values are indexed so as to support
        sequencing in descending order and also query by <, >, <=, >=, and =.
    """
        DIRECTION_UNSPECIFIED = 0
        ASCENDING = 1
        DESCENDING = 2
    direction = _messages.EnumField('DirectionValueValuesEnum', 1)
    name = _messages.StringField(2)