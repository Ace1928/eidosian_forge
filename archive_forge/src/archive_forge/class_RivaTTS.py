import asyncio
import logging
import pathlib
import queue
import tempfile
import threading
import wave
from enum import Enum
from typing import (
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import (
from langchain_core.runnables import RunnableConfig, RunnableSerializable
class RivaTTS(RivaAuthMixin, RivaCommonConfigMixin, RunnableSerializable[TTSInputType, TTSOutputType]):
    """A runnable that performs Text-to-Speech (TTS) with NVIDIA Riva."""
    name: str = 'nvidia_riva_tts'
    description: str = 'A tool for converting text to speech.This is useful for converting LLM output into audio bytes.'
    voice_name: str = Field('English-US.Female-1', description='The voice model in Riva to use for speech. Pre-trained models are documented in [the Riva documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html).')
    output_directory: Optional[str] = Field(None, description='The directory where all audio files should be saved. A null value indicates that wave files should not be saved. This is useful for debugging purposes.')

    @root_validator(pre=True)
    @classmethod
    def _validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the Python environment and input arguments."""
        _ = _import_riva_client()
        return values

    @validator('output_directory')
    @classmethod
    def _output_directory_validator(cls, v: str) -> str:
        if v:
            dirpath = pathlib.Path(v)
            dirpath.mkdir(parents=True, exist_ok=True)
            return str(dirpath.absolute())
        return v

    def _get_service(self) -> 'riva.client.SpeechSynthesisService':
        """Connect to the riva service and return the a client object."""
        riva_client = _import_riva_client()
        try:
            return riva_client.SpeechSynthesisService(self.auth)
        except Exception as err:
            raise ValueError('Error raised while connecting to the Riva TTS server.') from err

    def invoke(self, input: TTSInputType, _: Union[RunnableConfig, None]=None) -> TTSOutputType:
        """Perform TTS by taking a string and outputting the entire audio file."""
        return b''.join(self.transform(iter([input])))

    def transform(self, input: Iterator[TTSInputType], config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Iterator[TTSOutputType]:
        """Perform TTS by taking a stream of characters and streaming output bytes."""
        service = self._get_service()
        wav_file_name, wav_file = _mk_wave_file(self.output_directory, self.sample_rate_hertz)
        for chunk in _process_chunks(input):
            _LOGGER.debug('Riva TTS chunk: %s', chunk)
            responses = service.synthesize_online(text=chunk, voice_name=self.voice_name, language_code=self.language_code, encoding=self.encoding.riva_pb2, sample_rate_hz=self.sample_rate_hertz)
            for resp in responses:
                audio = cast(bytes, resp.audio)
                if wav_file:
                    wav_file.writeframesraw(audio)
                yield audio
        if wav_file:
            wav_file.close()
            _LOGGER.debug('Riva TTS wrote file: %s', wav_file_name)

    async def atransform(self, input: AsyncIterator[TTSInputType], config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncGenerator[TTSOutputType, None]:
        """Intercept async transforms and route them to the synchronous transform."""
        loop = asyncio.get_running_loop()
        input_queue: queue.Queue = queue.Queue()
        out_queue: asyncio.Queue = asyncio.Queue()

        async def _producer() -> None:
            """Produce input into the input queue."""
            async for val in input:
                input_queue.put_nowait(val)
            input_queue.put_nowait(_TRANSFORM_END)

        def _input_iterator() -> Iterator[TTSInputType]:
            """Iterate over the input_queue."""
            while True:
                try:
                    val = input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if val == _TRANSFORM_END:
                    break
                yield val

        def _consumer() -> None:
            """Consume the input with transform."""
            for val in self.transform(_input_iterator()):
                out_queue.put_nowait(val)
            out_queue.put_nowait(_TRANSFORM_END)

        async def _consumer_coro() -> None:
            """Coroutine that wraps the consumer."""
            await loop.run_in_executor(None, _consumer)
        producer = loop.create_task(_producer())
        consumer = loop.create_task(_consumer_coro())
        while True:
            try:
                val = await asyncio.wait_for(out_queue.get(), 0.5)
            except asyncio.exceptions.TimeoutError:
                continue
            out_queue.task_done()
            if val is _TRANSFORM_END:
                break
            yield val
        await producer
        await consumer