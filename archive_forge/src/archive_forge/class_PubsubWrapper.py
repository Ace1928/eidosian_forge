from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubWrapper(_messages.Message):
    """The payload to the push endpoint is in the form of the JSON
  representation of a PubsubMessage (https://cloud.google.com/pubsub/docs/refe
  rence/rpc/google.pubsub.v1#pubsubmessage).
  """