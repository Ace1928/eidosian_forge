from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepairActionValueValuesEnum(_messages.Enum):
    """Required. Repair action to take on specified resources of the node
    pool.

    Values:
      REPAIR_ACTION_UNSPECIFIED: No action will be taken by default.
      REPLACE: replace the specified list of nodes.
    """
    REPAIR_ACTION_UNSPECIFIED = 0
    REPLACE = 1