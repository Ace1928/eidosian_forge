from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsListRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsListRequest object.

  Fields:
    pageSize: Maximum number of subscriptions to return.
    pageToken: The value returned by the last `ListSubscriptionsResponse`;
      indicates that this is a continuation of a prior `ListSubscriptions`
      call, and that the system should return the next page of data.
    project: The name of the cloud project that subscriptions belong to.
      Format is `projects/{project}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    project = _messages.StringField(3, required=True)