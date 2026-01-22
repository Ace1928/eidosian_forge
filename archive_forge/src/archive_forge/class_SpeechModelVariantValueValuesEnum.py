from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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