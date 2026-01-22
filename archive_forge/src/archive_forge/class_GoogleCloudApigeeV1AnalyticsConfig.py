from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AnalyticsConfig(_messages.Message):
    """Configuration for the Analytics add-on.

  Enums:
    StateValueValuesEnum: Output only. The state of the Analytics add-on.

  Fields:
    enabled: Whether the Analytics add-on is enabled.
    expireTimeMillis: Output only. Time at which the Analytics add-on expires
      in milliseconds since epoch. If unspecified, the add-on will never
      expire.
    state: Output only. The state of the Analytics add-on.
    updateTime: Output only. The latest update time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the Analytics add-on.

    Values:
      ADDON_STATE_UNSPECIFIED: Default value.
      ENABLING: Add-on is in progress of enabling.
      ENABLED: Add-on is fully enabled and ready to use.
      DISABLING: Add-on is in progress of disabling.
      DISABLED: Add-on is fully disabled.
    """
        ADDON_STATE_UNSPECIFIED = 0
        ENABLING = 1
        ENABLED = 2
        DISABLING = 3
        DISABLED = 4
    enabled = _messages.BooleanField(1)
    expireTimeMillis = _messages.IntegerField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    updateTime = _messages.StringField(4)