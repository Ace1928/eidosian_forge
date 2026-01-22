from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1Intent(_messages.Message):
    """An intent categorizes an end-user's intention for one conversation turn.
  For each agent, you define many intents, where your combined intents can
  handle a complete conversation. When an end-user writes or says something,
  referred to as an end-user expression or end-user input, Dialogflow matches
  the end-user input to the best intent in your agent. Matching an intent is
  also known as intent classification. For more information, see the [intent
  guide](https://cloud.google.com/dialogflow/docs/intents-overview).

  Enums:
    DefaultResponsePlatformsValueListEntryValuesEnum:
    WebhookStateValueValuesEnum: Optional. Indicates whether webhooks are
      enabled for the intent.

  Fields:
    action: Optional. The name of the action associated with the intent. Note:
      The action name must not contain whitespaces.
    defaultResponsePlatforms: Optional. The list of platforms for which the
      first responses will be copied from the messages in PLATFORM_UNSPECIFIED
      (i.e. default platform).
    displayName: Required. The name of this intent.
    endInteraction: Optional. Indicates that this intent ends an interaction.
      Some integrations (e.g., Actions on Google or Dialogflow phone gateway)
      use this information to close interaction with an end user. Default is
      false.
    events: Optional. The collection of event names that trigger the intent.
      If the collection of input contexts is not empty, all of the contexts
      must be present in the active user session for an event to trigger this
      intent. Event names are limited to 150 characters.
    followupIntentInfo: Output only. Information about all followup intents
      that have this intent as a direct or indirect parent. We populate this
      field only in the output.
    inputContextNames: Optional. The list of context names required for this
      intent to be triggered. Formats: -
      `projects//agent/sessions/-/contexts/` -
      `projects//locations//agent/sessions/-/contexts/`
    isFallback: Optional. Indicates whether this is a fallback intent.
    liveAgentHandoff: Optional. Indicates that a live agent should be brought
      in to handle the interaction with the user. In most cases, when you set
      this flag to true, you would also want to set end_interaction to true as
      well. Default is false.
    messages: Optional. The collection of rich messages corresponding to the
      `Response` field in the Dialogflow console.
    mlDisabled: Optional. Indicates whether Machine Learning is disabled for
      the intent. Note: If `ml_disabled` setting is set to true, then this
      intent is not taken into account during inference in `ML ONLY` match
      mode. Also, auto-markup in the UI is turned off.
    mlEnabled: Optional. Indicates whether Machine Learning is enabled for the
      intent. Note: If `ml_enabled` setting is set to false, then this intent
      is not taken into account during inference in `ML ONLY` match mode.
      Also, auto-markup in the UI is turned off. DEPRECATED! Please use
      `ml_disabled` field instead. NOTE: If both `ml_enabled` and
      `ml_disabled` are either not set or false, then the default value is
      determined as follows: - Before April 15th, 2018 the default is:
      ml_enabled = false / ml_disabled = true. - After April 15th, 2018 the
      default is: ml_enabled = true / ml_disabled = false.
    name: Optional. The unique identifier of this intent. Required for
      Intents.UpdateIntent and Intents.BatchUpdateIntents methods. Supported
      formats: - `projects//agent/intents/` -
      `projects//locations//agent/intents/`
    outputContexts: Optional. The collection of contexts that are activated
      when the intent is matched. Context messages in this collection should
      not set the parameters field. Setting the `lifespan_count` to 0 will
      reset the context when the intent is matched. Format:
      `projects//agent/sessions/-/contexts/`.
    parameters: Optional. The collection of parameters associated with the
      intent.
    parentFollowupIntentName: Optional. The unique identifier of the parent
      intent in the chain of followup intents. You can set this field when
      creating an intent, for example with CreateIntent or BatchUpdateIntents,
      in order to make this intent a followup intent. It identifies the parent
      followup intent. Format: `projects//agent/intents/`.
    priority: Optional. The priority of this intent. Higher numbers represent
      higher priorities. - If the supplied value is unspecified or 0, the
      service translates the value to 500,000, which corresponds to the
      `Normal` priority in the console. - If the supplied value is negative,
      the intent is ignored in runtime detect intent requests.
    resetContexts: Optional. Indicates whether to delete all contexts in the
      current session when this intent is matched.
    rootFollowupIntentName: Output only. The unique identifier of the root
      intent in the chain of followup intents. It identifies the correct
      followup intents chain for this intent. Format:
      `projects//agent/intents/`.
    trainingPhrases: Optional. The collection of examples that the agent is
      trained on.
    webhookState: Optional. Indicates whether webhooks are enabled for the
      intent.
  """

    class DefaultResponsePlatformsValueListEntryValuesEnum(_messages.Enum):
        """DefaultResponsePlatformsValueListEntryValuesEnum enum type.

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

    class WebhookStateValueValuesEnum(_messages.Enum):
        """Optional. Indicates whether webhooks are enabled for the intent.

    Values:
      WEBHOOK_STATE_UNSPECIFIED: Webhook is disabled in the agent and in the
        intent.
      WEBHOOK_STATE_ENABLED: Webhook is enabled in the agent and in the
        intent.
      WEBHOOK_STATE_ENABLED_FOR_SLOT_FILLING: Webhook is enabled in the agent
        and in the intent. Also, each slot filling prompt is forwarded to the
        webhook.
    """
        WEBHOOK_STATE_UNSPECIFIED = 0
        WEBHOOK_STATE_ENABLED = 1
        WEBHOOK_STATE_ENABLED_FOR_SLOT_FILLING = 2
    action = _messages.StringField(1)
    defaultResponsePlatforms = _messages.EnumField('DefaultResponsePlatformsValueListEntryValuesEnum', 2, repeated=True)
    displayName = _messages.StringField(3)
    endInteraction = _messages.BooleanField(4)
    events = _messages.StringField(5, repeated=True)
    followupIntentInfo = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentFollowupIntentInfo', 6, repeated=True)
    inputContextNames = _messages.StringField(7, repeated=True)
    isFallback = _messages.BooleanField(8)
    liveAgentHandoff = _messages.BooleanField(9)
    messages = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessage', 10, repeated=True)
    mlDisabled = _messages.BooleanField(11)
    mlEnabled = _messages.BooleanField(12)
    name = _messages.StringField(13)
    outputContexts = _messages.MessageField('GoogleCloudDialogflowV2beta1Context', 14, repeated=True)
    parameters = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentParameter', 15, repeated=True)
    parentFollowupIntentName = _messages.StringField(16)
    priority = _messages.IntegerField(17, variant=_messages.Variant.INT32)
    resetContexts = _messages.BooleanField(18)
    rootFollowupIntentName = _messages.StringField(19)
    trainingPhrases = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentTrainingPhrase', 20, repeated=True)
    webhookState = _messages.EnumField('WebhookStateValueValuesEnum', 21)