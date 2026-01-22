from __future__ import annotations
from .speech import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .translations import (
from .transcriptions import (
class AsyncAudio(AsyncAPIResource):

    @cached_property
    def transcriptions(self) -> AsyncTranscriptions:
        return AsyncTranscriptions(self._client)

    @cached_property
    def translations(self) -> AsyncTranslations:
        return AsyncTranslations(self._client)

    @cached_property
    def speech(self) -> AsyncSpeech:
        return AsyncSpeech(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncAudioWithRawResponse:
        return AsyncAudioWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncAudioWithStreamingResponse:
        return AsyncAudioWithStreamingResponse(self)