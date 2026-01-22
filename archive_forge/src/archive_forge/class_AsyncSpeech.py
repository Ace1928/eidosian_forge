from __future__ import annotations
from typing import Union
from typing_extensions import Literal
import httpx
from ... import _legacy_response
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._utils import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import (
from ...types.audio import speech_create_params
from ..._base_client import (
class AsyncSpeech(AsyncAPIResource):

    @cached_property
    def with_raw_response(self) -> AsyncSpeechWithRawResponse:
        return AsyncSpeechWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncSpeechWithStreamingResponse:
        return AsyncSpeechWithStreamingResponse(self)

    async def create(self, *, input: str, model: Union[str, Literal['tts-1', 'tts-1-hd']], voice: Literal['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], response_format: Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'] | NotGiven=NOT_GIVEN, speed: float | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> _legacy_response.HttpxBinaryResponseContent:
        """
        Generates audio from the input text.

        Args:
          input: The text to generate audio for. The maximum length is 4096 characters.

          model:
              One of the available [TTS models](https://platform.openai.com/docs/models/tts):
              `tts-1` or `tts-1-hd`

          voice: The voice to use when generating the audio. Supported voices are `alloy`,
              `echo`, `fable`, `onyx`, `nova`, and `shimmer`. Previews of the voices are
              available in the
              [Text to speech guide](https://platform.openai.com/docs/guides/text-to-speech/voice-options).

          response_format: The format to audio in. Supported formats are `mp3`, `opus`, `aac`, `flac`,
              `wav`, and `pcm`.

          speed: The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is
              the default.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        extra_headers = {'Accept': 'application/octet-stream', **(extra_headers or {})}
        return await self._post('/audio/speech', body=await async_maybe_transform({'input': input, 'model': model, 'voice': voice, 'response_format': response_format, 'speed': speed}, speech_create_params.SpeechCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=_legacy_response.HttpxBinaryResponseContent)