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
def CreateRecognizer(self, resource, display_name, model, language_codes, profanity_filter=False, enable_word_time_offsets=False, enable_word_confidence=False, enable_automatic_punctuation=False, enable_spoken_punctuation=False, enable_spoken_emojis=False, min_speaker_count=None, max_speaker_count=None, encoding=None, sample_rate=None, audio_channel_count=None):
    """Call API CreateRecognizer method with provided arguments."""
    recognizer = self._messages.Recognizer(displayName=display_name, model=model, languageCodes=language_codes)
    recognizer.defaultRecognitionConfig = self._messages.RecognitionConfig()
    recognizer.defaultRecognitionConfig.features = self._messages.RecognitionFeatures()
    recognizer.defaultRecognitionConfig.features.profanityFilter = profanity_filter
    recognizer.defaultRecognitionConfig.features.enableWordTimeOffsets = enable_word_time_offsets
    recognizer.defaultRecognitionConfig.features.enableWordConfidence = enable_word_confidence
    recognizer.defaultRecognitionConfig.features.enableAutomaticPunctuation = enable_automatic_punctuation
    recognizer.defaultRecognitionConfig.features.enableSpokenPunctuation = enable_spoken_punctuation
    recognizer.defaultRecognitionConfig.features.enableSpokenEmojis = enable_spoken_emojis
    if min_speaker_count is not None and max_speaker_count is not None:
        recognizer.defaultRecognitionConfig.features.diarizationConfig = self._messages.SpeakerDiarizationConfig()
        recognizer.defaultRecognitionConfig.features.diarizationConfig.minSpeakerCount = min_speaker_count
        recognizer.defaultRecognitionConfig.features.diarizationConfig.maxSpeakerCount = max_speaker_count
    recognizer, _ = self._MatchEncoding(recognizer, encoding)
    if encoding is not None and encoding != 'AUTO':
        recognizer.defaultRecognitionConfig.explicitDecodingConfig.sampleRateHertz = sample_rate
        recognizer.defaultRecognitionConfig.explicitDecodingConfig.audioChannelCount = audio_channel_count
    request = self._messages.SpeechProjectsLocationsRecognizersCreateRequest(parent=resource.Parent(parent_collection='speech.projects.locations').RelativeName(), recognizerId=resource.Name(), recognizer=recognizer)
    return self._RecognizerServiceForLocation(location=resource.Parent().Name()).Create(request)