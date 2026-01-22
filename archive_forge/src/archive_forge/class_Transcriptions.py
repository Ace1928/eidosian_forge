from __future__ import annotations
from typing import List, Union, Mapping, cast
from typing_extensions import Literal
import httpx
from ... import _legacy_response
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from ..._utils import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ...types.audio import Transcription, transcription_create_params
from ..._base_client import (
class Transcriptions(SyncAPIResource):

    @cached_property
    def with_raw_response(self) -> TranscriptionsWithRawResponse:
        return TranscriptionsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> TranscriptionsWithStreamingResponse:
        return TranscriptionsWithStreamingResponse(self)

    def create(self, *, file: FileTypes, model: Union[str, Literal['whisper-1']], language: str | NotGiven=NOT_GIVEN, prompt: str | NotGiven=NOT_GIVEN, response_format: Literal['json', 'text', 'srt', 'verbose_json', 'vtt'] | NotGiven=NOT_GIVEN, temperature: float | NotGiven=NOT_GIVEN, timestamp_granularities: List[Literal['word', 'segment']] | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Transcription:
        """
        Transcribes audio into the input language.

        Args:
          file:
              The audio file object (not file name) to transcribe, in one of these formats:
              flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.

          model: ID of the model to use. Only `whisper-1` (which is powered by our open source
              Whisper V2 model) is currently available.

          language: The language of the input audio. Supplying the input language in
              [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will
              improve accuracy and latency.

          prompt: An optional text to guide the model's style or continue a previous audio
              segment. The
              [prompt](https://platform.openai.com/docs/guides/speech-to-text/prompting)
              should match the audio language.

          response_format: The format of the transcript output, in one of these options: `json`, `text`,
              `srt`, `verbose_json`, or `vtt`.

          temperature: The sampling temperature, between 0 and 1. Higher values like 0.8 will make the
              output more random, while lower values like 0.2 will make it more focused and
              deterministic. If set to 0, the model will use
              [log probability](https://en.wikipedia.org/wiki/Log_probability) to
              automatically increase the temperature until certain thresholds are hit.

          timestamp_granularities: The timestamp granularities to populate for this transcription.
              `response_format` must be set `verbose_json` to use timestamp granularities.
              Either or both of these options are supported: `word`, or `segment`. Note: There
              is no additional latency for segment timestamps, but generating word timestamps
              incurs additional latency.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        body = deepcopy_minimal({'file': file, 'model': model, 'language': language, 'prompt': prompt, 'response_format': response_format, 'temperature': temperature, 'timestamp_granularities': timestamp_granularities})
        files = extract_files(cast(Mapping[str, object], body), paths=[['file']])
        if files:
            extra_headers = {'Content-Type': 'multipart/form-data', **(extra_headers or {})}
        return self._post('/audio/transcriptions', body=maybe_transform(body, transcription_create_params.TranscriptionCreateParams), files=files, options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Transcription)