from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetStateValueValuesEnum(_messages.Enum):
    """Target instance state.

    Values:
      DELETED: The instance will be deleted.
      STOPPED: The instance will be stopped.
    """
    DELETED = 0
    STOPPED = 1