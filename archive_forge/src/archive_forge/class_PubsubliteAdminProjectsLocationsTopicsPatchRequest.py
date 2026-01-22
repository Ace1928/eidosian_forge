from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsTopicsPatchRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsTopicsPatchRequest object.

  Fields:
    name: The name of the topic. Structured like:
      projects/{project_number}/locations/{location}/topics/{topic_id}
    topic: A Topic resource to be passed as the request body.
    updateMask: Required. A mask specifying the topic fields to change.
  """
    name = _messages.StringField(1, required=True)
    topic = _messages.MessageField('Topic', 2)
    updateMask = _messages.StringField(3)