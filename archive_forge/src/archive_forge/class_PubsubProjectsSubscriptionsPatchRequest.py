from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsPatchRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsPatchRequest object.

  Fields:
    name: Required. The name of the subscription. It must have the format
      `"projects/{project}/subscriptions/{subscription}"`. `{subscription}`
      must start with a letter, and contain only letters (`[A-Za-z]`), numbers
      (`[0-9]`), dashes (`-`), underscores (`_`), periods (`.`), tildes (`~`),
      plus (`+`) or percent signs (`%`). It must be between 3 and 255
      characters in length, and it must not start with `"goog"`.
    updateSubscriptionRequest: A UpdateSubscriptionRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateSubscriptionRequest = _messages.MessageField('UpdateSubscriptionRequest', 2)