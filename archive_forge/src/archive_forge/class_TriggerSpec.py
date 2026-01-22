from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TriggerSpec(_messages.Message):
    """The desired state of the Trigger.

  Fields:
    broker: Broker is the broker that this trigger receives events from. If
      not provided, fully-managed Events for Cloud Run uses the `google`
      broker by default
    filter: Filter is the filter to apply against all events from the Broker.
      Only events that pass this filter will be sent to the Subscriber.
    subscriber: Subscriber is the addressable that receives events from the
      Broker that pass the Filter.
  """
    broker = _messages.StringField(1)
    filter = _messages.MessageField('TriggerFilter', 2)
    subscriber = _messages.MessageField('Destination', 3)