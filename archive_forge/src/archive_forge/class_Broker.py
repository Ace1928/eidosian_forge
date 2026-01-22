from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Broker(_messages.Message):
    """Broker collects a pool of events that are consumable using Triggers.
  Brokers provide a well-known endpoint for event delivery that senders can
  use with minimal knowledge of the event routing strategy. Subscribers use
  Triggers to request delivery of events from a Broker's pool to a specific
  URL or Addressable endpoint.

  Fields:
    apiVersion: The API version for this call.
    kind: The kind of resource, in this case always "Broker".
    metadata: Metadata associated with this Broker.
    spec: A BrokerSpec attribute.
    status: A BrokerStatus attribute.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    metadata = _messages.MessageField('ObjectMeta', 3)
    spec = _messages.MessageField('BrokerSpec', 4)
    status = _messages.MessageField('BrokerStatus', 5)