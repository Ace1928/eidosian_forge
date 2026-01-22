from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventSubscriptionDestination(_messages.Message):
    """Message for EventSubscription Destination to act on receiving an event

  Enums:
    TypeValueValuesEnum: type of the destination

  Fields:
    endpoint: OPTION 1: Hit an endpoint when we receive an event.
    serviceAccount: Service account needed for runtime plane to trigger IP
      workflow.
    type: type of the destination
  """

    class TypeValueValuesEnum(_messages.Enum):
        """type of the destination

    Values:
      TYPE_UNSPECIFIED: Default state.
      ENDPOINT: Endpoint - Hit the value of endpoint when event is received
    """
        TYPE_UNSPECIFIED = 0
        ENDPOINT = 1
    endpoint = _messages.MessageField('EndPoint', 1)
    serviceAccount = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)