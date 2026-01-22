from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolloutStatusValueValuesEnum(_messages.Enum):
    """Rollout status of the future quota limit.

    Values:
      IN_PROGRESS: IN_PROGRESS - A rollout is in process which will change the
        limit value to future limit.
      ROLLOUT_STATUS_UNSPECIFIED: ROLLOUT_STATUS_UNSPECIFIED - Rollout status
        is not specified. The default value.
    """
    IN_PROGRESS = 0
    ROLLOUT_STATUS_UNSPECIFIED = 1