from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsPullRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsPullRequest object.

  Fields:
    pullRequest: A PullRequest resource to be passed as the request body.
    subscription: The subscription from which messages should be pulled.
      Format is `projects/{project}/subscriptions/{sub}`.
  """
    pullRequest = _messages.MessageField('PullRequest', 1)
    subscription = _messages.StringField(2, required=True)