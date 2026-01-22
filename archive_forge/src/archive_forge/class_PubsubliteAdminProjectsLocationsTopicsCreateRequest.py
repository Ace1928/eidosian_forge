from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsTopicsCreateRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsTopicsCreateRequest object.

  Fields:
    parent: Required. The parent location in which to create the topic.
      Structured like `projects/{project_number}/locations/{location}`.
    topic: A Topic resource to be passed as the request body.
    topicId: Required. The ID to use for the topic, which will become the
      final component of the topic's name. This value is structured like: `my-
      topic-name`.
  """
    parent = _messages.StringField(1, required=True)
    topic = _messages.MessageField('Topic', 2)
    topicId = _messages.StringField(3)