from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SortOrderValueValuesEnum(_messages.Enum):
    """The sort order applied to the sort column.

    Values:
      SORT_ORDER_UNSPECIFIED: An unspecified sort order. This option is
        invalid when sorting is required.
      SORT_ORDER_NONE: No sorting is applied.
      SORT_ORDER_ASCENDING: The lowest-valued entries are selected first.
      SORT_ORDER_DESCENDING: The highest-valued entries are selected first.
    """
    SORT_ORDER_UNSPECIFIED = 0
    SORT_ORDER_NONE = 1
    SORT_ORDER_ASCENDING = 2
    SORT_ORDER_DESCENDING = 3