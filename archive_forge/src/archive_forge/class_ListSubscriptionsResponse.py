from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListSubscriptionsResponse(_messages.Message):
    """Response for the `ListSubscriptions` method.

  Fields:
    nextPageToken: If not empty, indicates that there may be more
      subscriptions that match the request; this value should be passed in a
      new `ListSubscriptionsRequest` to get more subscriptions.
    subscriptions: The subscriptions that match the request.
  """
    nextPageToken = _messages.StringField(1)
    subscriptions = _messages.MessageField('Subscription', 2, repeated=True)