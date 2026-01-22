from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1WebhookRequestFulfillmentInfo(_messages.Message):
    """Represents fulfillment information communicated to the webhook.

  Fields:
    tag: Always present. The value of the Fulfillment.tag field will be
      populated in this field by Dialogflow when the associated webhook is
      called. The tag is typically used by the webhook service to identify
      which fulfillment is being called, but it could be used for other
      purposes.
  """
    tag = _messages.StringField(1)