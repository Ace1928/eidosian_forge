from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginTypeValueValuesEnum(_messages.Enum):
    """The origin of this inventory item.

    Values:
      ORIGIN_TYPE_UNSPECIFIED: Invalid. An origin type must be specified.
      INVENTORY_REPORT: This inventory item was discovered as the result of
        the agent reporting inventory via the reporting API.
    """
    ORIGIN_TYPE_UNSPECIFIED = 0
    INVENTORY_REPORT = 1