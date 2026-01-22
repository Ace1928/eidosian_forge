from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KafkaSource(_messages.Message):
    """Kafka Source configuration.

  Fields:
    brokerUris: Required. The Kafka broker URIs. e.g. 10.12.34.56:8080
    consumerGroupId: Required. The consumer group ID used by the Kafka broker
      to track the offsets of all topic partitions being read by this Stream.
    kafkaAuthenticationConfig: Optional. Authentication configuration used to
      authenticate the Kafka client with the Kafka broker, and authorize to
      read the topic(s).
    topics: Required. The Kafka topics to read from.
  """
    brokerUris = _messages.StringField(1, repeated=True)
    consumerGroupId = _messages.StringField(2)
    kafkaAuthenticationConfig = _messages.MessageField('KafkaAuthenticationConfig', 3)
    topics = _messages.StringField(4, repeated=True)