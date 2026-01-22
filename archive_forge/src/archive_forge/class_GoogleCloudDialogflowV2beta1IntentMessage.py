from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessage(_messages.Message):
    """Corresponds to the `Response` field in the Dialogflow console.

  Enums:
    PlatformValueValuesEnum: Optional. The platform that this message is
      intended for.

  Messages:
    PayloadValue: A custom platform-specific response.

  Fields:
    basicCard: Displays a basic card for Actions on Google.
    browseCarouselCard: Browse carousel card for Actions on Google.
    card: Displays a card.
    carouselSelect: Displays a carousel card for Actions on Google.
    image: Displays an image.
    linkOutSuggestion: Displays a link out suggestion chip for Actions on
      Google.
    listSelect: Displays a list card for Actions on Google.
    mediaContent: The media content card for Actions on Google.
    payload: A custom platform-specific response.
    platform: Optional. The platform that this message is intended for.
    quickReplies: Displays quick replies.
    rbmCarouselRichCard: Rich Business Messaging (RBM) carousel rich card
      response.
    rbmStandaloneRichCard: Standalone Rich Business Messaging (RBM) rich card
      response.
    rbmText: Rich Business Messaging (RBM) text response. RBM allows
      businesses to send enriched and branded versions of SMS. See
      https://jibe.google.com/business-messaging.
    simpleResponses: Returns a voice or text-only response for Actions on
      Google.
    suggestions: Displays suggestion chips for Actions on Google.
    tableCard: Table card for Actions on Google.
    telephonyPlayAudio: Plays audio from a file in Telephony Gateway.
    telephonySynthesizeSpeech: Synthesizes speech in Telephony Gateway.
    telephonyTransferCall: Transfers the call in Telephony Gateway.
    text: Returns a text response.
  """

    class PlatformValueValuesEnum(_messages.Enum):
        """Optional. The platform that this message is intended for.

    Values:
      PLATFORM_UNSPECIFIED: Not specified.
      FACEBOOK: Facebook.
      SLACK: Slack.
      TELEGRAM: Telegram.
      KIK: Kik.
      SKYPE: Skype.
      LINE: Line.
      VIBER: Viber.
      ACTIONS_ON_GOOGLE: Google Assistant See [Dialogflow webhook format](http
        s://developers.google.com/assistant/actions/build/json/dialogflow-
        webhook-json)
      TELEPHONY: Telephony Gateway.
      GOOGLE_HANGOUTS: Google Hangouts.
    """
        PLATFORM_UNSPECIFIED = 0
        FACEBOOK = 1
        SLACK = 2
        TELEGRAM = 3
        KIK = 4
        SKYPE = 5
        LINE = 6
        VIBER = 7
        ACTIONS_ON_GOOGLE = 8
        TELEPHONY = 9
        GOOGLE_HANGOUTS = 10

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PayloadValue(_messages.Message):
        """A custom platform-specific response.

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
    basicCard = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageBasicCard', 1)
    browseCarouselCard = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageBrowseCarouselCard', 2)
    card = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageCard', 3)
    carouselSelect = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageCarouselSelect', 4)
    image = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageImage', 5)
    linkOutSuggestion = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageLinkOutSuggestion', 6)
    listSelect = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageListSelect', 7)
    mediaContent = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageMediaContent', 8)
    payload = _messages.MessageField('PayloadValue', 9)
    platform = _messages.EnumField('PlatformValueValuesEnum', 10)
    quickReplies = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageQuickReplies', 11)
    rbmCarouselRichCard = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmCarouselCard', 12)
    rbmStandaloneRichCard = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmStandaloneCard', 13)
    rbmText = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmText', 14)
    simpleResponses = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageSimpleResponses', 15)
    suggestions = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageSuggestions', 16)
    tableCard = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageTableCard', 17)
    telephonyPlayAudio = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageTelephonyPlayAudio', 18)
    telephonySynthesizeSpeech = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageTelephonySynthesizeSpeech', 19)
    telephonyTransferCall = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageTelephonyTransferCall', 20)
    text = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageText', 21)