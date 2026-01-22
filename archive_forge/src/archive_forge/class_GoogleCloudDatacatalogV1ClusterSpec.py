from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ClusterSpec(_messages.Message):
    """Additional specification of a cluster.

  Fields:
    kafkaCluster: Fields specific to a Kafka cluster. Present only on
      corresponding Kafka cluster entries.
  """
    kafkaCluster = _messages.MessageField('GoogleCloudDatacatalogV1KafkaClusterSpec', 1)