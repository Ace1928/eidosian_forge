from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TieringPolicy(_messages.Message):
    """Defines tiering policy for the volume.

  Enums:
    TierActionValueValuesEnum: Optional. Flag indicating if the volume has
      tiering policy enable/pause. Default is PAUSED.

  Fields:
    coolingThresholdDays: Optional. Time in days to mark the volume's data
      block as cold and make it eligible for tiering, can be range from 7-183.
      Default is 31.
    tierAction: Optional. Flag indicating if the volume has tiering policy
      enable/pause. Default is PAUSED.
  """

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
    coolingThresholdDays = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    tierAction = _messages.EnumField('TierActionValueValuesEnum', 2)