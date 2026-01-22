from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2QueryParameters(_messages.Message):
    """Represents the parameters of the conversational query.

  Messages:
    PayloadValue: This field can be used to pass custom data to your webhook.
      Arbitrary JSON objects are supported. If supplied, the value is used to
      populate the `WebhookRequest.original_detect_intent_request.payload`
      field sent to your webhook.
    WebhookHeadersValue: This field can be used to pass HTTP headers for a
      webhook call. These headers will be sent to webhook along with the
      headers that have been configured through the Dialogflow web console.
      The headers defined within this field will overwrite the headers
      configured through the Dialogflow console if there is a conflict. Header
      names are case-insensitive. Google's specified headers are not allowed.
      Including: "Host", "Content-Length", "Connection", "From", "User-Agent",
      "Accept-Encoding", "If-Modified-Since", "If-None-Match", "X-Forwarded-
      For", etc.

  Fields:
    contexts: The collection of contexts to be activated before this query is
      executed.
    geoLocation: The geo location of this conversational query.
    payload: This field can be used to pass custom data to your webhook.
      Arbitrary JSON objects are supported. If supplied, the value is used to
      populate the `WebhookRequest.original_detect_intent_request.payload`
      field sent to your webhook.
    platform: The platform of the virtual agent response messages. If not
      empty, only emits messages from this platform in the response. Valid
      values are the enum names of platform.
    resetContexts: Specifies whether to delete all contexts in the current
      session before the new ones are activated.
    sentimentAnalysisRequestConfig: Configures the type of sentiment analysis
      to perform. If not provided, sentiment analysis is not performed.
    sessionEntityTypes: Additional session entity types to replace or extend
      developer entity types with. The entity synonyms apply to all languages
      and persist for the session of this query.
    timeZone: The time zone of this conversational query from the [time zone
      database](https://www.iana.org/time-zones), e.g., America/New_York,
      Europe/Paris. If not provided, the time zone specified in agent settings
      is used.
    webhookHeaders: This field can be used to pass HTTP headers for a webhook
      call. These headers will be sent to webhook along with the headers that
      have been configured through the Dialogflow web console. The headers
      defined within this field will overwrite the headers configured through
      the Dialogflow console if there is a conflict. Header names are case-
      insensitive. Google's specified headers are not allowed. Including:
      "Host", "Content-Length", "Connection", "From", "User-Agent", "Accept-
      Encoding", "If-Modified-Since", "If-None-Match", "X-Forwarded-For", etc.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PayloadValue(_messages.Message):
        """This field can be used to pass custom data to your webhook. Arbitrary
    JSON objects are supported. If supplied, the value is used to populate the
    `WebhookRequest.original_detect_intent_request.payload` field sent to your
    webhook.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class WebhookHeadersValue(_messages.Message):
        """This field can be used to pass HTTP headers for a webhook call. These
    headers will be sent to webhook along with the headers that have been
    configured through the Dialogflow web console. The headers defined within
    this field will overwrite the headers configured through the Dialogflow
    console if there is a conflict. Header names are case-insensitive.
    Google's specified headers are not allowed. Including: "Host", "Content-
    Length", "Connection", "From", "User-Agent", "Accept-Encoding", "If-
    Modified-Since", "If-None-Match", "X-Forwarded-For", etc.

    Messages:
      AdditionalProperty: An additional property for a WebhookHeadersValue
        object.

    Fields:
      additionalProperties: Additional properties of type WebhookHeadersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a WebhookHeadersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    contexts = _messages.MessageField('GoogleCloudDialogflowV2Context', 1, repeated=True)
    geoLocation = _messages.MessageField('GoogleTypeLatLng', 2)
    payload = _messages.MessageField('PayloadValue', 3)
    platform = _messages.StringField(4)
    resetContexts = _messages.BooleanField(5)
    sentimentAnalysisRequestConfig = _messages.MessageField('GoogleCloudDialogflowV2SentimentAnalysisRequestConfig', 6)
    sessionEntityTypes = _messages.MessageField('GoogleCloudDialogflowV2SessionEntityType', 7, repeated=True)
    timeZone = _messages.StringField(8)
    webhookHeaders = _messages.MessageField('WebhookHeadersValue', 9)