from __future__ import annotations
from .speech import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .translations import (
from .transcriptions import (
class AudioWithRawResponse:

    def __init__(self, audio: Audio) -> None:
        self._audio = audio

    @cached_property
    def transcriptions(self) -> TranscriptionsWithRawResponse:
        return TranscriptionsWithRawResponse(self._audio.transcriptions)

    @cached_property
    def translations(self) -> TranslationsWithRawResponse:
        return TranslationsWithRawResponse(self._audio.translations)

    @cached_property
    def speech(self) -> SpeechWithRawResponse:
        return SpeechWithRawResponse(self._audio.speech)