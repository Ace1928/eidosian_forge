from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsDeleteRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsDeleteRequest object.

  Fields:
    subscription: The subscription to delete. Format is
      `projects/{project}/subscriptions/{sub}`.
  """
    subscription = _messages.StringField(1, required=True)