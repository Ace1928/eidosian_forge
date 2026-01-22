from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from six.moves import urllib
def UpdateRecognizer(self, resource, display_name=None, model=None, language_codes=None, profanity_filter=None, enable_word_time_offsets=None, enable_word_confidence=None, enable_automatic_punctuation=None, enable_spoken_punctuation=None, enable_spoken_emojis=None, min_speaker_count=None, max_speaker_count=None, encoding=None, sample_rate=None, audio_channel_count=None):
    """Call API UpdateRecognizer method with provided arguments."""
    recognizer = self._messages.Recognizer()
    update_mask = []
    if display_name is not None:
        recognizer.displayName = display_name
        update_mask.append('display_name')
    if model is not None:
        recognizer.model = model
        update_mask.append('model')
    if language_codes is not None:
        recognizer.languageCodes = language_codes
        update_mask.append('language_codes')
    if recognizer.defaultRecognitionConfig is None:
        recognizer.defaultRecognitionConfig = self._messages.RecognitionConfig()
    if recognizer.defaultRecognitionConfig.features is None:
        recognizer.defaultRecognitionConfig.features = self._messages.RecognitionFeatures()
    features = recognizer.defaultRecognitionConfig.features
    if profanity_filter is not None:
        features.profanityFilter = profanity_filter
        update_mask.append('default_recognition_config.features.profanity_filter')
    if enable_word_time_offsets is not None:
        features.enableWordTimeOffsets = enable_word_time_offsets
        update_mask.append('default_recognition_config.features.enable_word_time_offsets')
    if enable_word_confidence is not None:
        features.enableWordConfidence = enable_word_confidence
        update_mask.append('default_recognition_config.features.enable_word_confidence')
    if enable_automatic_punctuation is not None:
        features.enableAutomaticPunctuation = enable_automatic_punctuation
        update_mask.append('default_recognition_config.features.enable_automatic_punctuation')
    if enable_spoken_punctuation is not None:
        features.enableSpokenPunctuation = enable_spoken_punctuation
        update_mask.append('default_recognition_config.features.enable_spoken_punctuation')
    if enable_spoken_emojis is not None:
        features.enableSpokenEmojis = enable_spoken_emojis
        update_mask.append('default_recognition_config.features.enable_spoken_emojis')
    if features.diarizationConfig is None and (min_speaker_count is not None or max_speaker_count is not None):
        features.diarizationConfig = self._messages.SpeakerDiarizationConfig()
    if min_speaker_count is not None:
        features.diarizationConfig.minSpeakerCount = min_speaker_count
        update_mask.append('default_recognition_config.features.diarization_config.min_speaker_count')
    if max_speaker_count is not None:
        features.diarizationConfig.maxSpeakerCount = max_speaker_count
        update_mask.append('default_recognition_config.features.diarization_config.max_speaker_count')
    recognizer, update_mask = self._MatchEncoding(recognizer, encoding, update=True, update_mask=update_mask)
    if sample_rate is not None:
        if recognizer.defaultRecognitionConfig.explicitDecodingConfig is None:
            recognizer.defaultRecognitionConfig.explicitDecodingConfig = self._messages.ExplicitDecodingConfig()
        recognizer.defaultRecognitionConfig.explicitDecodingConfig.sampleRateHertz = sample_rate
        update_mask.append('default_recognition_config.explicit_decoding_config.sample_rate_hertz')
    if audio_channel_count is not None:
        if recognizer.defaultRecognitionConfig.explicitDecodingConfig is None:
            recognizer.defaultRecognitionConfig.explicitDecodingConfig = self._messages.ExplicitDecodingConfig()
        recognizer.defaultRecognitionConfig.explicitDecodingConfig.audioChannelCount = audio_channel_count
        update_mask.append('default_recognition_config.explicit_decoding_config.audio_channel_count')
    request = self._messages.SpeechProjectsLocationsRecognizersPatchRequest(name=resource.RelativeName(), recognizer=recognizer, updateMask=','.join(update_mask))
    return self._RecognizerServiceForLocation(location=resource.Parent().Name()).Patch(request)