from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaintenanceStatusValueValuesEnum(_messages.Enum):
    """MaintenanceStatusValueValuesEnum enum type.

    Values:
      ONGOING: There is ongoing maintenance on this VM.
      PENDING: There is pending maintenance.
      UNKNOWN: Unknown maintenance status. Do not use this value.
    """
    ONGOING = 0
    PENDING = 1
    UNKNOWN = 2