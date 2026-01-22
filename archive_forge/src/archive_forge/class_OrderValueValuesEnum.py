from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrderValueValuesEnum(_messages.Enum):
    """Indicates that this field supports ordering by the specified order or
    comparing using =, !=, <, <=, >, >=.

    Values:
      ORDER_UNSPECIFIED: The ordering is unspecified. Not a valid option.
      ASCENDING: The field is ordered by ascending field value.
      DESCENDING: The field is ordered by descending field value.
    """
    ORDER_UNSPECIFIED = 0
    ASCENDING = 1
    DESCENDING = 2