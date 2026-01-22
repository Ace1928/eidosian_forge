from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubSubAvroFormat(_messages.Message):
    """Configuration for reading Cloud Storage data written via [Cloud Storage
  subscriptions](https://cloud.google.com/pubsub/docs/cloudstorage). The data
  and attributes fields of the originally exported Pub/Sub message will be
  restored when publishing.
  """