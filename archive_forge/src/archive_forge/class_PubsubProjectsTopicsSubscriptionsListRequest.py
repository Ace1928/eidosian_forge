from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsTopicsSubscriptionsListRequest(_messages.Message):
    """A PubsubProjectsTopicsSubscriptionsListRequest object.

  Fields:
    pageSize: Maximum number of subscription names to return.
    pageToken: The value returned by the last
      `ListTopicSubscriptionsResponse`; indicates that this is a continuation
      of a prior `ListTopicSubscriptions` call, and that the system should
      return the next page of data.
    topic: The name of the topic that subscriptions are attached to. Format is
      `projects/{project}/topics/{topic}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    topic = _messages.StringField(3, required=True)