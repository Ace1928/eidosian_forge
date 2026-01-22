from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubDestination(_messages.Message):
    """A Pub/Sub destination.

  Fields:
    topic: The name of the Pub/Sub topic to publish to. Example:
      `projects/PROJECT_ID/topics/TOPIC_ID`.
  """
    topic = _messages.StringField(1)