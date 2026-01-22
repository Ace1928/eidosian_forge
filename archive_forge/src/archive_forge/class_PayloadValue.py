from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PayloadValue(_messages.Message):
    """Optional. This field can be used to pass custom data from your webhook
    to the integration or API caller. Arbitrary JSON objects are supported.
    When provided, Dialogflow uses this field to populate
    QueryResult.webhook_payload sent to the integration or API caller. This
    field is also used by the [Google Assistant
    integration](https://cloud.google.com/dialogflow/docs/integrations/aog)
    for rich response messages. See the format definition at [Google Assistant
    Dialogflow webhook format](https://developers.google.com/assistant/actions
    /build/json/dialogflow-webhook-json)

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