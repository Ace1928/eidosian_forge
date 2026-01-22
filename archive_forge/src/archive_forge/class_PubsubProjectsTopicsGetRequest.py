from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsTopicsGetRequest(_messages.Message):
    """A PubsubProjectsTopicsGetRequest object.

  Fields:
    topic: The name of the topic to get. Format is
      `projects/{project}/topics/{topic}`.
  """
    topic = _messages.StringField(1, required=True)