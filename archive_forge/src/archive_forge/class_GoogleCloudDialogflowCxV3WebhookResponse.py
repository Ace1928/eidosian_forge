from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3WebhookResponse(_messages.Message):
    """The response message for a webhook call.

  Messages:
    PayloadValue: Value to append directly to QueryResult.webhook_payloads.

  Fields:
    fulfillmentResponse: The fulfillment response to send to the user. This
      field can be omitted by the webhook if it does not intend to send any
      response to the user.
    pageInfo: Information about page status. This field can be omitted by the
      webhook if it does not intend to modify page status.
    payload: Value to append directly to QueryResult.webhook_payloads.
    sessionInfo: Information about session status. This field can be omitted
      by the webhook if it does not intend to modify session status.
    targetFlow: The target flow to transition to. Format:
      `projects//locations//agents//flows/`.
    targetPage: The target page to transition to. Format:
      `projects//locations//agents//flows//pages/`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PayloadValue(_messages.Message):
        """Value to append directly to QueryResult.webhook_payloads.

    Messages:
      AdditionalProperty: An additional property for a PayloadValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PayloadValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    fulfillmentResponse = _messages.MessageField('GoogleCloudDialogflowCxV3WebhookResponseFulfillmentResponse', 1)
    pageInfo = _messages.MessageField('GoogleCloudDialogflowCxV3PageInfo', 2)
    payload = _messages.MessageField('PayloadValue', 3)
    sessionInfo = _messages.MessageField('GoogleCloudDialogflowCxV3SessionInfo', 4)
    targetFlow = _messages.StringField(5)
    targetPage = _messages.StringField(6)