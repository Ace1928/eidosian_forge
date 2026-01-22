from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DataStreamSpec(_messages.Message):
    """Additional specification of a data stream.

  Fields:
    kafkaTopic: Fields specific to a Kafka topic. Present only on
      corresponding Kafka topic entries.
  """
    kafkaTopic = _messages.MessageField('GoogleCloudDatacatalogV1KafkaTopicSpec', 1)