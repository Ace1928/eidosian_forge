from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ConversationProfile(_messages.Message):
    """Defines the services to connect to incoming Dialogflow conversations.

  Fields:
    automatedAgentConfig: Configuration for an automated agent to use with
      this profile.
    createTime: Output only. Create time of the conversation profile.
    displayName: Required. Human readable name for this profile. Max length
      1024 bytes.
    humanAgentAssistantConfig: Configuration for agent assistance to use with
      this profile.
    humanAgentHandoffConfig: Configuration for connecting to a live agent.
      Currently, this feature is not general available, please contact Google
      to get access.
    languageCode: Language code for the conversation profile. If not
      specified, the language is en-US. Language at ConversationProfile should
      be set for all non en-US languages. This should be a
      [BCP-47](https://www.rfc-editor.org/rfc/bcp/bcp47.txt) language tag.
      Example: "en-US".
    loggingConfig: Configuration for logging conversation lifecycle events.
    name: The unique identifier of this conversation profile. Format:
      `projects//locations//conversationProfiles/`.
    newMessageEventNotificationConfig: Configuration for publishing new
      message events. Event will be sent in format of ConversationEvent
    notificationConfig: Configuration for publishing conversation lifecycle
      events.
    securitySettings: Name of the CX SecuritySettings reference for the agent.
      Format: `projects//locations//securitySettings/`.
    sttConfig: Settings for speech transcription.
    timeZone: The time zone of this conversational profile from the [time zone
      database](https://www.iana.org/time-zones), e.g., America/New_York,
      Europe/Paris. Defaults to America/New_York.
    ttsConfig: Configuration for Text-to-Speech synthesization. Used by Phone
      Gateway to specify synthesization options. If agent defines
      synthesization options as well, agent settings overrides the option
      here.
    updateTime: Output only. Update time of the conversation profile.
  """
    automatedAgentConfig = _messages.MessageField('GoogleCloudDialogflowV2AutomatedAgentConfig', 1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    humanAgentAssistantConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfig', 4)
    humanAgentHandoffConfig = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentHandoffConfig', 5)
    languageCode = _messages.StringField(6)
    loggingConfig = _messages.MessageField('GoogleCloudDialogflowV2LoggingConfig', 7)
    name = _messages.StringField(8)
    newMessageEventNotificationConfig = _messages.MessageField('GoogleCloudDialogflowV2NotificationConfig', 9)
    notificationConfig = _messages.MessageField('GoogleCloudDialogflowV2NotificationConfig', 10)
    securitySettings = _messages.StringField(11)
    sttConfig = _messages.MessageField('GoogleCloudDialogflowV2SpeechToTextConfig', 12)
    timeZone = _messages.StringField(13)
    ttsConfig = _messages.MessageField('GoogleCloudDialogflowV2SynthesizeSpeechConfig', 14)
    updateTime = _messages.StringField(15)