from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubscriptionInfo(_messages.Message):
    """Message containing information for the appliance subscription.

  Enums:
    UpdateTypeValueValuesEnum: Input only. The time when the update will be
      applied. Only needed for an update.

  Fields:
    effectiveTime: Input only. The timestamp when update will be appliance if
      update_type is SPECIFIC_DATE.
    subscriptionTerm: Required. Input only. Customer selected subscription
      term.
    subscriptions: Optional. List of subscription details.
    updateType: Input only. The time when the update will be applied. Only
      needed for an update.
  """

    class UpdateTypeValueValuesEnum(_messages.Enum):
        """Input only. The time when the update will be applied. Only needed for
    an update.

    Values:
      CHANGE_TIME_TYPE_UNSPECIFIED: Default value. Should not be used.
      END_OF_PERIOD: End of the period.
      END_OF_TERM: End of the term.
      SPECIFIC_DATE: At a specific date.
      EARLIEST_POSSIBLE: At the earliest time possible.
    """
        CHANGE_TIME_TYPE_UNSPECIFIED = 0
        END_OF_PERIOD = 1
        END_OF_TERM = 2
        SPECIFIC_DATE = 3
        EARLIEST_POSSIBLE = 4
    effectiveTime = _messages.StringField(1)
    subscriptionTerm = _messages.StringField(2)
    subscriptions = _messages.MessageField('Subscription', 3, repeated=True)
    updateType = _messages.EnumField('UpdateTypeValueValuesEnum', 4)