from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ViewTypeValueValuesEnum(_messages.Enum):
    """Type of this view.

    Values:
      VIEW_TYPE_UNSPECIFIED: Default unknown view type.
      STANDARD_VIEW: Standard view.
      MATERIALIZED_VIEW: Materialized view.
    """
    VIEW_TYPE_UNSPECIFIED = 0
    STANDARD_VIEW = 1
    MATERIALIZED_VIEW = 2