from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolloutStrategyValueValuesEnum(_messages.Enum):
    """Endpoints rollout strategy. If FIXED, config_id must be specified. If
    MANAGED, config_id must be omitted.

    Values:
      UNSPECIFIED_ROLLOUT_STRATEGY: Not specified. Defaults to FIXED.
      FIXED: Endpoints service configuration ID will be fixed to the
        configuration ID specified by config_id.
      MANAGED: Endpoints service configuration ID will be updated with each
        rollout.
    """
    UNSPECIFIED_ROLLOUT_STRATEGY = 0
    FIXED = 1
    MANAGED = 2