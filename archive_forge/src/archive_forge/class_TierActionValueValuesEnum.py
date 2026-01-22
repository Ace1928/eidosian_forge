from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TierActionValueValuesEnum(_messages.Enum):
    """Optional. Flag indicating if the volume has tiering policy
    enable/pause. Default is PAUSED.

    Values:
      TIER_ACTION_UNSPECIFIED: Unspecified.
      ENABLED: When tiering is enabled, new cold data will be tiered.
      PAUSED: When paused, tiering won't be performed on new data. Existing
        data stays tiered until accessed.
    """
    TIER_ACTION_UNSPECIFIED = 0
    ENABLED = 1
    PAUSED = 2