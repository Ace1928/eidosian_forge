from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsAcknowledgeRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsAcknowledgeRequest object.

  Fields:
    acknowledgeRequest: A AcknowledgeRequest resource to be passed as the
      request body.
    subscription: The subscription whose message is being acknowledged. Format
      is `projects/{project}/subscriptions/{sub}`.
  """
    acknowledgeRequest = _messages.MessageField('AcknowledgeRequest', 1)
    subscription = _messages.StringField(2, required=True)