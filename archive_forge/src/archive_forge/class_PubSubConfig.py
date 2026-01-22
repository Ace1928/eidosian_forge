from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubSubConfig(_messages.Message):
    """Configuration for exporting to a Pub/Sub topic.

  Fields:
    topic: The name of the Pub/Sub topic. Structured like:
      projects/{project_number}/topics/{topic_id}. The topic may be changed.
  """
    topic = _messages.StringField(1)