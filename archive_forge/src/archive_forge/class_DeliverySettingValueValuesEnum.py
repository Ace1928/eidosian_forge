from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeliverySettingValueValuesEnum(_messages.Enum):
    """Output only. Delivery setting associated with the membership.

    Values:
      DELIVERY_SETTING_UNSPECIFIED: Default. Should not be used.
      ALL_MAIL: Represents each mail should be delivered
      DIGEST: Represents 1 email for every 25 messages.
      DAILY: Represents daily summary of messages.
      NONE: Represents no delivery.
      DISABLED: Represents disabled state.
    """
    DELIVERY_SETTING_UNSPECIFIED = 0
    ALL_MAIL = 1
    DIGEST = 2
    DAILY = 3
    NONE = 4
    DISABLED = 5