from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2InputAudioConfig(_messages.Message):
    """Instructs the speech recognizer how to process the audio content.

  Enums:
    AudioEncodingValueValuesEnum: Required. Audio encoding of the audio
      content to process.
    ModelVariantValueValuesEnum: Which variant of the Speech model to use.

  Fields:
    audioEncoding: Required. Audio encoding of the audio content to process.
    disableNoSpeechRecognizedEvent: Only used in Participants.AnalyzeContent
      and Participants.StreamingAnalyzeContent. If `false` and recognition
      doesn't return any result, trigger `NO_SPEECH_RECOGNIZED` event to
      Dialogflow agent.
    enableAutomaticPunctuation: Enable automatic punctuation option at the
      speech backend.
    enableWordInfo: If `true`, Dialogflow returns SpeechWordInfo in
      StreamingRecognitionResult with information about the recognized speech
      words, e.g. start and end time offsets. If false or unspecified, Speech
      doesn't return any word-level information.
    languageCode: Required. The language of the supplied audio. Dialogflow
      does not do translations. See [Language
      Support](https://cloud.google.com/dialogflow/docs/reference/language)
      for a list of the currently supported language codes. Note that queries
      in the same session do not necessarily need to specify the same
      language.
    model: Optional. Which Speech model to select for the given request. For
      more information, see [Speech
      models](https://cloud.google.com/dialogflow/es/docs/speech-models).
    modelVariant: Which variant of the Speech model to use.
    optOutConformerModelMigration: If `true`, the request will opt out for STT
      conformer model migration. This field will be deprecated once force
      migration takes place in June 2024. Please refer to [Dialogflow ES
      Speech model
      migration](https://cloud.google.com/dialogflow/es/docs/speech-model-
      migration).
    phraseHints: A list of strings containing words and phrases that the
      speech recognizer should recognize with higher likelihood. See [the
      Cloud Speech documentation](https://cloud.google.com/speech-to-
      text/docs/basics#phrase-hints) for more details. This field is
      deprecated. Please use [`speech_contexts`]() instead. If you specify
      both [`phrase_hints`]() and [`speech_contexts`](), Dialogflow will treat
      the [`phrase_hints`]() as a single additional [`SpeechContext`]().
    sampleRateHertz: Required. Sample rate (in Hertz) of the audio content
      sent in the query. Refer to [Cloud Speech API
      documentation](https://cloud.google.com/speech-to-text/docs/basics) for
      more details.
    singleUtterance: If `false` (default), recognition does not cease until
      the client closes the stream. If `true`, the recognizer will detect a
      single spoken utterance in input audio. Recognition ceases when it
      detects the audio's voice has stopped or paused. In this case, once a
      detected intent is received, the client should close the stream and
      start a new request with a new stream as needed. Note: This setting is
      relevant only for streaming methods. Note: When specified,
      InputAudioConfig.single_utterance takes precedence over
      StreamingDetectIntentRequest.single_utterance.
    speechContexts: Context information to assist speech recognition. See [the
      Cloud Speech documentation](https://cloud.google.com/speech-to-
      text/docs/basics#phrase-hints) for more details.
  """

    class AudioEncodingValueValuesEnum(_messages.Enum):
        """Required. Audio encoding of the audio content to process.

    Values:
      AUDIO_ENCODING_UNSPECIFIED: Not specified.
      AUDIO_ENCODING_LINEAR_16: Uncompressed 16-bit signed little-endian
        samples (Linear PCM).
      AUDIO_ENCODING_FLAC: [`FLAC`](https://xiph.org/flac/documentation.html)
        (Free Lossless Audio Codec) is the recommended encoding because it is
        lossless (therefore recognition is not compromised) and requires only
        about half the bandwidth of `LINEAR16`. `FLAC` stream encoding
        supports 16-bit and 24-bit samples, however, not all fields in
        `STREAMINFO` are supported.
      AUDIO_ENCODING_MULAW: 8-bit samples that compand 14-bit audio samples
        using G.711 PCMU/mu-law.
      AUDIO_ENCODING_AMR: Adaptive Multi-Rate Narrowband codec.
        `sample_rate_hertz` must be 8000.
      AUDIO_ENCODING_AMR_WB: Adaptive Multi-Rate Wideband codec.
        `sample_rate_hertz` must be 16000.
      AUDIO_ENCODING_OGG_OPUS: Opus encoded audio frames in Ogg container
        ([OggOpus](https://wiki.xiph.org/OggOpus)). `sample_rate_hertz` must
        be 16000.
      AUDIO_ENCODING_SPEEX_WITH_HEADER_BYTE: Although the use of lossy
        encodings is not recommended, if a very low bitrate encoding is
        required, `OGG_OPUS` is highly preferred over Speex encoding. The
        [Speex](https://speex.org/) encoding supported by Dialogflow API has a
        header byte in each block, as in MIME type `audio/x-speex-with-header-
        byte`. It is a variant of the RTP Speex encoding defined in [RFC
        5574](https://tools.ietf.org/html/rfc5574). The stream is a sequence
        of blocks, one block per RTP packet. Each block starts with a byte
        containing the length of the block, in bytes, followed by one or more
        frames of Speex data, padded to an integral number of bytes (octets)
        as specified in RFC 5574. In other words, each RTP header is replaced
        with a single byte containing the block length. Only Speex wideband is
        supported. `sample_rate_hertz` must be 16000.
    """
        AUDIO_ENCODING_UNSPECIFIED = 0
        AUDIO_ENCODING_LINEAR_16 = 1
        AUDIO_ENCODING_FLAC = 2
        AUDIO_ENCODING_MULAW = 3
        AUDIO_ENCODING_AMR = 4
        AUDIO_ENCODING_AMR_WB = 5
        AUDIO_ENCODING_OGG_OPUS = 6
        AUDIO_ENCODING_SPEEX_WITH_HEADER_BYTE = 7

    class ModelVariantValueValuesEnum(_messages.Enum):
        """Which variant of the Speech model to use.

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
    audioEncoding = _messages.EnumField('AudioEncodingValueValuesEnum', 1)
    disableNoSpeechRecognizedEvent = _messages.BooleanField(2)
    enableAutomaticPunctuation = _messages.BooleanField(3)
    enableWordInfo = _messages.BooleanField(4)
    languageCode = _messages.StringField(5)
    model = _messages.StringField(6)
    modelVariant = _messages.EnumField('ModelVariantValueValuesEnum', 7)
    optOutConformerModelMigration = _messages.BooleanField(8)
    phraseHints = _messages.StringField(9, repeated=True)
    sampleRateHertz = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    singleUtterance = _messages.BooleanField(11)
    speechContexts = _messages.MessageField('GoogleCloudDialogflowV2SpeechContext', 12, repeated=True)