from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SpeechToTextConfig(_messages.Message):
    """Configures speech transcription for ConversationProfile.

  Enums:
    SpeechModelVariantValueValuesEnum: The speech model used in speech to
      text. `SPEECH_MODEL_VARIANT_UNSPECIFIED`, `USE_BEST_AVAILABLE` will be
      treated as `USE_ENHANCED`. It can be overridden in AnalyzeContentRequest
      and StreamingAnalyzeContentRequest request. If enhanced model variant is
      specified and an enhanced version of the specified model for the
      language does not exist, then it would emit an error.

  Fields:
    model: Which Speech model to select. Select the model best suited to your
      domain to get best results. If a model is not explicitly specified, then
      Dialogflow auto-selects a model based on other parameters in the
      SpeechToTextConfig and Agent settings. If enhanced speech model is
      enabled for the agent and an enhanced version of the specified model for
      the language does not exist, then the speech is recognized using the
      standard version of the specified model. Refer to [Cloud Speech API
      documentation](https://cloud.google.com/speech-to-
      text/docs/basics#select-model) for more details. If you specify a model,
      the following models typically have the best performance: - phone_call
      (best for Agent Assist and telephony) - latest_short (best for
      Dialogflow non-telephony) - command_and_search Leave this field
      unspecified to use [Agent Speech settings](https://cloud.google.com/dial
      ogflow/cx/docs/concept/agent#settings-speech) for model selection.
    speechModelVariant: The speech model used in speech to text.
      `SPEECH_MODEL_VARIANT_UNSPECIFIED`, `USE_BEST_AVAILABLE` will be treated
      as `USE_ENHANCED`. It can be overridden in AnalyzeContentRequest and
      StreamingAnalyzeContentRequest request. If enhanced model variant is
      specified and an enhanced version of the specified model for the
      language does not exist, then it would emit an error.
    useTimeoutBasedEndpointing: Use timeout based endpointing, interpreting
      endpointer sensitivy as seconds of timeout value.
  """

    class SpeechModelVariantValueValuesEnum(_messages.Enum):
        """The speech model used in speech to text.
    `SPEECH_MODEL_VARIANT_UNSPECIFIED`, `USE_BEST_AVAILABLE` will be treated
    as `USE_ENHANCED`. It can be overridden in AnalyzeContentRequest and
    StreamingAnalyzeContentRequest request. If enhanced model variant is
    specified and an enhanced version of the specified model for the language
    does not exist, then it would emit an error.

    Values:
      SPEECH_MODEL_VARIANT_UNSPECIFIED: No model variant specified. In this
        case Dialogflow defaults to USE_BEST_AVAILABLE.
      USE_BEST_AVAILABLE: Use the best available variant of the Speech model
        that the caller is eligible for. Please see the [Dialogflow
        docs](https://cloud.google.com/dialogflow/docs/data-logging) for how
        to make your project eligible for enhanced models.
      USE_STANDARD: Use standard model variant even if an enhanced model is
        available. See the [Cloud Speech
        documentation](https://cloud.google.com/speech-to-text/docs/enhanced-
        models) for details about enhanced models.
      USE_ENHANCED: Use an enhanced model variant: * If an enhanced variant
        does not exist for the given model and request language, Dialogflow
        falls back to the standard variant. The [Cloud Speech
        documentation](https://cloud.google.com/speech-to-text/docs/enhanced-
        models) describes which models have enhanced variants. * If the API
        caller isn't eligible for enhanced models, Dialogflow returns an
        error. Please see the [Dialogflow
        docs](https://cloud.google.com/dialogflow/docs/data-logging) for how
        to make your project eligible.
    """
        SPEECH_MODEL_VARIANT_UNSPECIFIED = 0
        USE_BEST_AVAILABLE = 1
        USE_STANDARD = 2
        USE_ENHANCED = 3
    model = _messages.StringField(1)
    speechModelVariant = _messages.EnumField('SpeechModelVariantValueValuesEnum', 2)
    useTimeoutBasedEndpointing = _messages.BooleanField(3)