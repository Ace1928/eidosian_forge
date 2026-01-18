import asyncio  # For asynchronous operations to improve responsiveness and concurrency.
import logging  # For detailed logging of events and debugging.
import time  # For time-related functions, such as measuring intervals.
from collections import (
    deque,
)  # Efficient double-ended queue for managing audio buffers.
from typing import (
    AsyncGenerator,
    List,
    Optional,
    Generator,
    AsyncIterator,
)  # Type hinting for better code readability and maintainability.

import numpy as np  # For numerical operations on audio data.
import psutil  # For monitoring system resources to adapt processing.
import pyttsx3  # For text-to-speech functionality.
import sounddevice as sd  # For capturing audio input from the microphone.
import whisper  # For speech-to-text transcription.
from llama_index.core import PromptTemplate  # For creating prompts for the LLM.
from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
    AgentChatResponse,
)  # For handling streaming chat responses from the LLM.
from llama_index.core.llms import ChatMessage, ChatResponse  # For LLM chat messages.
from rich.console import Console  # For enhanced console output.
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import noisereduce as nr  # For noise reduction in audio.
from queue import Queue  # Import Queue from queue
from threading import Event  # Import Event from threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from accelerate import disk_offload

from eidos_basellm import (
    BaseLLM,
)  # Assuming this is the LLM implementation.
from eidos_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_HF_TOKEN,
)  # Importing default configurations.

# Configure detailed logging for comprehensive debugging and monitoring.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(filename)s:%(lineno)d - %(message)s",
)

# Initialize Rich Console for formatted output.
console = Console()
logging.info("Initializing Rich Console.")
# Initialize the LLM globally.
logging.info(f"Initializing LLM with model name: {DEFAULT_MODEL_NAME}")
eidos = BaseLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct")

logging.info("LLM initialized.")

# Initialize the TTS engine. Done once for efficiency.
logging.info("Initializing TTS engine...")
tts_engine = pyttsx3.init()  # Initialize the pyttsx3 engine.
tts_engine.setProperty("rate", 175)  # Set a natural speaking rate.
logging.debug(f"TTS engine rate set to: {tts_engine.getProperty('rate')}")
voices = tts_engine.getProperty("voices")  # Get available voices.
if voices:
    tts_engine.setProperty("voice", voices[0].id)  # Set to the first available voice.
    logging.info(f"TTS voice set to: {voices[0].id}")
else:
    logging.warning("No TTS voices found.")


class VoiceActivityDetector:
    """Detects voice activity using dynamic thresholding for improved accuracy."""

    def __init__(
        self,
        initial_threshold: float = 0.003,  # Initial threshold for speech detection.
        history_length: int = 10,  # Number of recent RMS values to track for dynamic adjustment.
        noise_level_adjust_speed: float = 0.01,  # Rate at which the noise level adapts.
        min_rms_for_speech: float = 0.005,  # Minimum RMS value to consider as potential speech.
        dynamic_adjust_range: float = 0.2,  # Range within which to dynamically adjust the threshold.
    ):
        """Initializes the VAD with parameters for dynamic thresholding."""
        self.initial_threshold = (
            initial_threshold  # Store initial_threshold as an instance attribute.
        )
        self.threshold = initial_threshold  # Current threshold for speech detection.
        self.history_length = history_length  # Number of recent RMS values to track.
        self.rms_history = deque(maxlen=history_length)  # Store recent RMS values.
        self._lock = asyncio.Lock()  # Asynchronous lock for thread safety.
        self.noise_level = initial_threshold  # Initial estimate of the noise level.
        self.noise_level_adjust_speed = (
            noise_level_adjust_speed  # Rate of noise level adaptation.
        )
        self.min_rms_for_speech = (
            min_rms_for_speech  # Minimum RMS to avoid overly sensitive detection.
        )
        self.dynamic_adjust_range = (
            dynamic_adjust_range  # Range for dynamic threshold adjustments.
        )
        logging.info(
            f"VAD initialized with threshold: {initial_threshold}, history length: {history_length}, adjust speed: {noise_level_adjust_speed}, dynamic range: {dynamic_adjust_range}"
        )

    async def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Checks if the current audio chunk contains speech using dynamic thresholding."""
        async with self._lock:
            # Calculate the Root Mean Square (RMS) of the audio chunk.
            rms = np.sqrt(np.mean(audio_chunk**2))
            self.rms_history.append(rms)  # Add the current RMS value to the history.
            logging.debug(f"VAD - Current RMS: {rms:.5f}")

            # Dynamically adjust the noise level and threshold based on recent RMS history.
            if len(self.rms_history) == self.history_length:
                avg_rms = np.mean(list(self.rms_history))
                max_rms = np.max(self.rms_history)

                # Adjust noise level based on the average RMS.
                self.noise_level = (
                    self.noise_level * (1 - self.noise_level_adjust_speed)
                    + avg_rms * self.noise_level_adjust_speed
                )
                # Calculate a dynamic threshold based on the noise level and a range.
                self.threshold = self.noise_level + self.dynamic_adjust_range
                # Ensure the threshold doesn't go below the initial threshold or minimum RMS.
                self.threshold = max(
                    self.initial_threshold,
                    self.threshold,
                    self.min_rms_for_speech * 1.2,
                )  # Added a small buffer to min_rms

                logging.debug(
                    f"VAD - Avg RMS: {avg_rms:.5f}, Max RMS: {max_rms:.5f}, Adjusted noise level: {self.noise_level:.5f}, New threshold: {self.threshold:.5f}"
                )

            # Consider it speech if the RMS is significantly above the threshold.
            is_speaking = rms > self.threshold
            logging.debug(
                f"VAD - RMS: {rms:.5f}, Threshold: {self.threshold:.5f}, Speech detected: {is_speaking}"
            )
            return is_speaking  # Return True if speech is detected, False otherwise.


# Load the Whisper model globally for efficiency.
logging.info("Loading Whisper model...")
stt = disk_offload(
    model=(whisper.load_model(name="tiny.en", device="cpu")),
    execution_device="cpu",  # type: ignore
    offload_dir="whisper_offload_cache",
    offload_buffers=True,
)  # Load a smaller Whisper model for faster processing.
logging.info("Whisper model loaded.")

# Define the prompt template for the LLM.
from llama_index.core.prompts import PromptTemplate

DEFAULT_PROMPT = PromptTemplate(
    "You are a helpful assistant. Respond to the user's query.\n\n"
    "Chat History:\n{history}\n\n"
    "User Query:\n{input}\n\n"
    "Response:"
)


class InterruptibleStreamCompleteIterator(AsyncIterator[ChatResponse]):
    """Wraps a stream iterator to allow interruption."""

    def __init__(self, stream_iterator: AsyncGenerator[ChatResponse, None]):
        """Initializes with an asynchronous stream iterator."""
        self.stream_iterator = stream_iterator  # The asynchronous generator to wrap.
        self._interrupted = (
            False  # Flag to indicate if the stream has been interrupted.
        )
        self._lock = asyncio.Lock()  # Asynchronous lock for thread-safe access.
        logging.debug("InterruptibleStreamCompleteIterator created.")

    def __aiter__(self):
        """Returns the asynchronous iterator."""
        return self  # Return self to be used in async for loops.

    async def __anext__(self):
        """Returns the next item or raises StopAsyncIteration if interrupted."""
        async with self._lock:
            if self._interrupted:
                logging.debug("Stream interrupted, raising StopAsyncIteration.")
                raise StopAsyncIteration  # Signal that the iteration should stop.
            try:
                return await self.stream_iterator.__anext__()  # Get the next item.
            except StopAsyncIteration:
                raise  # Re-raise if the underlying iterator is exhausted.
            except Exception as e:
                logging.error(f"Error in stream iterator: {e}")
                raise  # Raise any other exceptions encountered.

    async def interrupt(self):
        """Sets the interrupted flag."""
        async with self._lock:
            self._interrupted = True  # Set the interrupt flag.
            logging.debug("Interrupt flag set for stream.")


async def sync_to_async_generator(sync_generator):
    """Converts a synchronous generator to an asynchronous generator."""
    for item in sync_generator:
        yield item


async def get_llm_response(
    text: str,
    chat_history: Optional[List[ChatMessage]] = None,
    interrupt_event: Optional[asyncio.Event] = None,
) -> StreamingAgentChatResponse:
    """Generates a streaming response from the LLM."""
    logging.info(f"LLM - Getting response for text: '{text[:50]}...'")
    chat_history = chat_history or []  # Initialize chat history if None is provided.

    # Format the prompt using the chat history and the current input text.
    formatted_prompt = DEFAULT_PROMPT.format(
        history="\n".join(
            [f"{msg.role}: {msg.content}" for msg in chat_history[-20:]]
        ),  # Use the last 5 messages for context.
        input=text,
    )
    logging.debug(f"LLM - Formatted prompt: '{formatted_prompt[:150]}...'")

    # Get a streaming response from the LLM.
    logging.info("LLM - Starting stream complete.")
    response_stream = eidos.stream_complete(formatted_prompt)
    # Wrap the stream in an interruptible iterator.
    interruptible_iterator = InterruptibleStreamCompleteIterator(
        sync_to_async_generator(response_stream)
    )

    if interrupt_event:
        interrupt_event.clear()  # Ensure the interrupt event is not set initially.

    async def response_generator() -> Optional[AsyncGenerator[ChatResponse, None]]:
        """Asynchronously generates LLM responses and handles interruptions."""
        logging.debug("LLM - Starting response generator.")
        try:
            async for (
                r
            ) in (
                interruptible_iterator
            ):  # Iterate through the LLM's streaming response.
                logging.debug(f"LLM - Received response chunk: {r.delta}")
                if (
                    interrupt_event and interrupt_event.is_set()
                ):  # Check for interruption.
                    logging.info("LLM - Response generation interrupted.")
                    break  # Stop generating if interrupted.
                if r.delta:
                    yield ChatResponse(
                        message=ChatMessage(role="assistant", content=r.delta)
                    )  # Yield the delta as a ChatMessage.
        except StopAsyncIteration:
            logging.debug("LLM - Response stream completed.")
        except Exception as e:
            logging.error(f"LLM - Response generator error: {e}")
        finally:
            await interruptible_iterator.interrupt()  # Ensure the iterator is interrupted on exit.
            logging.debug("LLM - Response generator finished.")

    return StreamingAgentChatResponse(
        achat_stream=await response_generator(),  # The asynchronous generator for the response.
        aqueue=None,  # Async queue (not directly used).
        response="",  # Initial empty response.
        sources=[],  # No sources initially.
        source_nodes=[],  # No source nodes initially.
        unformatted_response="",  # Initial empty unformatted response.
        queue=Queue(),  # Queue for managing response chunks (not directly used in this streaming setup).
        is_function=False,  # Not a function call.
        new_item_event=None,  # No new item event.
        is_function_false_event=None,  # No function false event.
        is_function_not_none_thread_event=Event(),  # Event for function calls (not used here).
        is_writing_to_memory=True,  # Indicates the response should be written to memory/history.
        exception=None,  # No initial exception.
    )


class AudioProcessor:
    """Processes audio, detects voice activity, and manages transcription with low latency."""

    def __init__(
        self,
        vad_initial_threshold: float = 0.003,  # Initial threshold for voice activity detection.
        chunk_size: int = 256,  # Shorter chunk size for lower latency.
        min_speech_duration: float = 0.1,  # Minimum duration of speech to consider.
        min_silence_duration: float = 0.2,  # Minimum duration of silence to end a speech segment.
        sample_rate: int = 16000,  # Sampling rate of the audio.
        noise_reduce_factor: float = 0.7,  # Factor for noise reduction.
        overlap_count: int = 128,  # Number of overlapping samples.
        vad_history_length: int = 5,  # History length for VAD smoothing.
        vad_dynamic_adjust_range: float = 0.1,  # Range for dynamic VAD threshold adjustment.
    ):
        """Initializes the AudioProcessor."""
        self.sample_rate = sample_rate  # Sampling rate of the audio.
        self.chunk_size = chunk_size  # Size of audio chunks to process.
        self.overlap_count = overlap_count  # Number of overlapping samples.
        self.vad = VoiceActivityDetector(
            initial_threshold=vad_initial_threshold,
            history_length=vad_history_length,
            dynamic_adjust_range=vad_dynamic_adjust_range,
        )  # Initialize voice activity detector.
        self.transcription_queue = (
            asyncio.Queue()
        )  # Queue to hold transcribed text chunks.
        self.stop_event = (
            asyncio.Event()
        )  # Event to signal the audio processing to stop.
        self.console = Console()  # Rich console for output.
        self.audio_buffer = deque(
            maxlen=int(sample_rate * 5)
        )  # Ring buffer for the last few seconds of audio.
        self.min_speech_duration = min_speech_duration  # Minimum duration of speech.
        self.min_silence_duration = min_silence_duration  # Minimum duration of silence.
        self.speech_start_time = None  # Time when speech started.
        self.silence_start_time = None  # Time when silence started.
        self._lock = asyncio.Lock()  # Lock for thread-safe access to shared resources.
        self._executor = ThreadPoolExecutor()
        self._is_speaking = False  # Flag to indicate if speech is currently detected.
        self._resource_check_interval = 0.5  # Interval to check system resources.
        self._last_resource_check = (
            asyncio.get_event_loop().time()
        )  # Last time resources were checked.
        self._ongoing_transcription = (
            False  # Flag to prevent concurrent transcriptions.
        )
        self.noise_reduce_factor = noise_reduce_factor  # Factor for noise reduction.
        self.previous_chunk = np.zeros(
            chunk_size, dtype=np.int16
        )  # Store the previous chunk for overlap.
        logging.info("AudioProcessor initialized.")
        self.microphone_active = False
        self.audio_queue = (
            asyncio.Queue()
        )  # Queue for passing audio chunks from callback to async processors.
        self._high_load = False  # Flag to indicate high system load
        self._high_load_start_time = None  # Time when high load was detected
        self._high_load_cooldown = 5  # Cooldown period after high load
        self._is_running = False  # Flag to indicate if the audio processor is running

    def _audio_callback(self, indata, frames, time, status):
        """Callback for the audio stream."""
        if status:
            self.console.print(f"Error from audio stream: {status}", style="bold red")
            logging.error(f"Error from audio stream: {status}")
        try:
            if indata is None or len(indata) == 0 or not self.microphone_active:
                return
            current_chunk = np.frombuffer(indata, dtype=np.int16).flatten()
            # Combine with previous chunk for overlap
            audio_chunk_with_overlap = np.concatenate(
                [self.previous_chunk[-self.overlap_count :], current_chunk]
            )
            self.previous_chunk = current_chunk.copy()
            # Instead of processing directly here, put the chunk on a queue.
            asyncio.run_coroutine_threadsafe(
                self.audio_queue.put(audio_chunk_with_overlap),
                asyncio.get_event_loop(),
            )
        except Exception as e:
            logging.error(f"Error in audio callback: {e}", exc_info=True)

    async def process_audio_chunks(self):
        """Continuously processes audio chunks from the queue."""
        while self._is_running and not self.stop_event.is_set():
            try:
                audio_chunk = await self.audio_queue.get()
                await self._process_audio_chunk(audio_chunk)
            except Exception as e:
                logging.error(
                    f"Error processing queued audio chunk: {e}", exc_info=True
                )

    async def start(self):
        """Starts the audio processing."""
        if self._is_running:
            logging.warning("Audio processor is already running.")
            return
        self.stop_event.clear()  # Ensure the stop event is cleared before starting.
        logging.info("Starting audio stream...")
        self.microphone_active = True
        self._is_running = True
        # Start the asynchronous task for processing audio chunks.
        asyncio.create_task(self.process_audio_chunks())
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,  # Set the sampling rate.
                blocksize=self.chunk_size,  # Set the block size for audio chunks.
                channels=1,  # Mono audio.
                dtype="int16",  # Data type of the audio stream.
                callback=self._audio_callback,  # Callback function for audio data.
                latency="low",
            ):
                await self.process_audio_for_transcription()  # Start processing audio.
        except Exception as e:
            logging.error(f"Error in audio stream: {e}", exc_info=True)
        finally:
            logging.info("Audio stream closed.")
            self.microphone_active = False
            self._is_running = False

    async def process_audio_chunk(self, raw_chunk: np.ndarray):
        """Processes individual audio chunks."""
        try:
            # Removed asyncio.create_task
            pass
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}", exc_info=True)

    async def _process_audio_chunk(self, raw_chunk: np.ndarray):
        """Asynchronously processes audio data for VAD and transcription."""
        chunk_float32 = (
            raw_chunk.astype(np.float32) / 32768.0
        )  # Normalize the audio chunk to float32.
        rms = np.sqrt(np.mean(chunk_float32**2))
        logging.debug(f"Audio - Received audio chunk, RMS: {rms:.5f}")

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
            y=chunk_float32,
            sr=self.sample_rate,
            stationary=False,
            prop_decrease=self.noise_reduce_factor,
        )

        is_currently_speech = await self.vad.is_speech(
            reduced_noise
        )  # Detect voice activity.

        async with self._lock:
            if is_currently_speech:
                if not self._is_speaking:
                    self._is_speaking = True  # Mark that speech has started.
                    self.speech_start_time = (
                        asyncio.get_event_loop().time()
                    )  # Record the start time of speech.
                    self.silence_start_time = None  # Reset the silence start time.
                    logging.info("Audio - Speech started.")
                self.audio_buffer.extend(
                    raw_chunk.flatten().tolist()
                )  # Append raw chunk for better transcription
            else:
                if self._is_speaking:
                    if self.silence_start_time is None:
                        self.silence_start_time = (
                            asyncio.get_event_loop().time()
                        )  # Record silence start.
                        logging.info("Audio - Silence started.")
                    elif (
                        asyncio.get_event_loop().time() - self.silence_start_time
                    ) >= self.min_silence_duration:
                        # Consider speech segment complete if silence exceeds threshold.
                        if (
                            self.speech_start_time is not None
                            and (
                                asyncio.get_event_loop().time() - self.speech_start_time
                            )
                            >= self.min_speech_duration
                            and not self._ongoing_transcription
                        ):
                            self._ongoing_transcription = True
                            audio_segment_np = (
                                np.array(self.audio_buffer).astype(np.float32) / 32768.0
                            )
                            if audio_segment_np.size > 0:
                                logging.info("Audio - Submitting transcription task.")
                                asyncio.create_task(
                                    self._transcribe_segment(audio_segment_np)
                                )
                            self.audio_buffer.clear()
                            self._is_speaking = False
                            self.speech_start_time = None
                            self.silence_start_time = None
                            logging.info(
                                "Audio - Speech ended, transcription triggered."
                            )
                elif (
                    self.audio_buffer
                    and not self._is_speaking
                    and not self._ongoing_transcription
                ):
                    # Transcribe remaining buffer if no speech detected (for short utterances).
                    self._ongoing_transcription = True
                    audio_segment_np = (
                        np.array(self.audio_buffer).astype(np.float32) / 32768.0
                    )
                    if audio_segment_np.size > 0:
                        logging.info("Audio - Transcribing remaining buffer.")
                        asyncio.create_task(self._transcribe_segment(audio_segment_np))
                    self.audio_buffer.clear()

    async def get_transcription_chunk(
        self, timeout: Optional[float] = None
    ) -> Optional[str]:
        """Retrieves a transcribed text chunk from the queue."""
        try:
            logging.debug("Transcription - Waiting for transcription chunk.")
            chunk = await asyncio.wait_for(
                self.transcription_queue.get(), timeout=timeout
            )
            logging.debug(f"Transcription - Received chunk: '{chunk}'")
            return chunk
        except asyncio.TimeoutError:
            logging.debug("Transcription - Timeout waiting for chunk.")
            return None

    async def _transcribe_segment(self, audio_segment: np.ndarray):
        """Transcribes the given audio segment using Whisper."""
        logging.info("Transcription - Starting transcription of audio segment.")
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                self._executor, transcribe, audio_segment
            )
            if text.strip():
                await self.transcription_queue.put(text)
                logging.info(f"Transcription - Transcribed text: '{text}'")
            else:
                logging.info("Transcription - No text transcribed from segment.")
        except Exception as e:
            logging.error(f"Transcription - Error: {e}", exc_info=True)
        finally:
            self._ongoing_transcription = False
            logging.info("Transcription - Transcription finished.")

    async def process_audio_for_transcription(self):
        """Manages audio processing, voice activity detection, and transcription."""
        logging.info("Starting audio processing for transcription...")
        while self._is_running and not self.stop_event.is_set():
            try:
                if (
                    asyncio.get_event_loop().time() - self._last_resource_check
                ) > self._resource_check_interval:
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent
                    logging.debug(
                        f"System - CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%"
                    )
                    if cpu_usage > 90 or memory_usage > 90:
                        if not self._high_load:
                            self._high_load = True
                            self._high_load_start_time = asyncio.get_event_loop().time()
                            self.microphone_active = False
                            logging.warning(
                                "System - High load detected, pausing audio input."
                            )
                        elif (
                            self._high_load_start_time is not None
                            and (
                                asyncio.get_event_loop().time()
                                - self._high_load_start_time
                            )
                            > self._high_load_cooldown
                        ):
                            self._high_load = False
                            self.microphone_active = True
                            logging.info("System - Load reduced, resuming audio input.")
                    elif self._high_load and self._high_load_start_time is not None:
                        if (
                            asyncio.get_event_loop().time() - self._high_load_start_time
                        ) > self._high_load_cooldown:
                            self._high_load = False
                            self.microphone_active = True
                            logging.info("System - Load reduced, resuming audio input.")
                    self._last_resource_check = asyncio.get_event_loop().time()
                await asyncio.sleep(0.001)
            except Exception as e:
                logging.error(f"Error in audio processing loop: {e}", exc_info=True)
        logging.info("Stopped audio processing for transcription.")

    async def stop(self):
        """Stops the audio processing."""
        self.stop_event.set()
        logging.info("Stopping audio processing...")


def transcribe(audio_np: np.ndarray) -> str:
    """Transcribes audio using the Whisper model."""
    try:
        logging.debug("Whisper - Starting transcription.")
        result = stt.transcribe(audio_np)
        logging.debug(f"Whisper - Result: {result}")
        if isinstance(result["text"], list):
            text = " ".join(result["text"])
        else:
            text = result["text"]
        logging.info(f"Whisper - Transcribed text: '{text}'")
        return text.strip()
    except Exception as e:
        logging.error(f"Whisper - Transcription failed: {e}", exc_info=True)
        return ""


async def play_audio(text_chunk: str, tts_stop_event: asyncio.Event):
    """Plays audio using pyttsx3 with interruption capability."""
    if not text_chunk.strip():
        return

    tts_stop_event.clear()  # Clear the stop event before starting playback.

    def on_word():
        """Callback function for when a word starts being spoken."""
        if tts_stop_event.is_set():
            logging.debug("TTS - Stop event detected, stopping...")
            tts_engine.stop()  # Stop the TTS engine immediately.

    def on_end(name, completed):
        """Callback function for when an utterance is finished."""
        logging.debug(
            f"TTS - Finished speaking utterance: {name}, completed: {completed}"
        )
        pass  # No action needed when an utterance completes normally.

    try:
        tts_engine.connect(
            "started-word", on_word
        )  # Connect the 'started-word' signal to the on_word callback.
        tts_engine.connect(
            "finished-utterance", on_end
        )  # Connect the 'finished-utterance' signal to the on_end callback.
        logging.info(f"TTS - Saying: '{text_chunk}'")
        tts_engine.say(text_chunk)  # Queue the text for speech.
        tts_engine.runAndWait()  # Block until all queued text is spoken or stopped.
    except Exception as e:
        logging.error(f"TTS - Error: {e}", exc_info=True)
    finally:
        try:
            tts_engine.disconnect("started-word")  # Disconnect the signal.
            tts_engine.disconnect("finished-utterance")  # Disconnect the signal.
        except Exception as e:
            logging.warning(f"TTS - Error disconnecting signals: {e}", exc_info=True)
        logging.debug("TTS - Finished processing.")


def stop_audio():
    """Immediately stops the TTS engine."""
    logging.info("Stopping TTS immediately.")
    tts_engine.stop()


"""Manages the main conversation flow, including starting and stopping audio processing."""

try:
    with Live(console=console, screen=True, refresh_per_second=10) as live:

        record_button = "[bold green][+][/bold green] Record"
        stop_button = "[bold red][-][/bold red] Stop"
        status_text = Text("Ready to record", style="bold blue")
        audio_processor = AudioProcessor()
        llm_processing_lock = asyncio.Lock()
        microphone_on = True

        async def update_display():
            table = Table(title="Eidos Talk Status")
            table.add_column("Component")
            table.add_column("Status")
            table.add_column("Info")

            table.add_row(
                "Microphone",
                ("🟢 Active" if audio_processor.microphone_active else "🔴 Inactive"),
                "",
            )
            table.add_row(
                "Voice Detection",
                "👂 Listening" if audio_processor._is_speaking else " silence",
                f"Threshold: {audio_processor.vad.threshold:.4f}",
            )
            table.add_row(
                "Transcription",
                "📝 Processing" if audio_processor._ongoing_transcription else "Idle",
                "",
            )
            table.add_row(
                "LLM", "🧠 Thinking" if llm_processing_lock.locked() else "Idle", ""
            )
            live.update(Panel(table))
            await asyncio.sleep(0.1)

        async def record_and_process():
            global status_text
            global microphone_on
            if not microphone_on:
                status_text = Text("Recording...", style="bold green")
                microphone_on = True
                await audio_processor.start()
            else:
                status_text = Text("Processing...", style="bold yellow")
                microphone_on = False
                await audio_processor.stop()
            status_text = Text("Ready to record", style="bold blue")

        async def main_loop():
            global status_text
            while True:
                await update_display()
                user_input = (
                    console.input(f"{record_button} / {stop_button}").strip().lower()
                )
                if user_input == "+":
                    await record_and_process()
                elif user_input == "-":
                    if microphone_on:
                        await record_and_process()
                await asyncio.sleep(0.1)

        asyncio.run(main_loop())
except KeyboardInterrupt:
    console.print("\n[red]Exiting...")
finally:
    asyncio.run(audio_processor.stop())
