from email import generator
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

import asyncio
import logging
import nest_asyncio
import psutil
import threading
import time
from typing import (
    Dict,
    Tuple,
)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    TextIteratorStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from llama_index.core.llms import (
    ChatMessage,
    MessageRole,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    CustomLLM,
)
from eidos_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_INITIAL_MAX_TOKENS,
    DEFAULT_DEVICE_MAP,
    DEFAULT_TRUST_REMOTE_CODE,
    DEFAULT_HIGH_RESOURCE_THRESHOLD,
    DEFAULT_INITIAL_CHUNK_SIZE,
    DEFAULT_SYSTEM_PROMPT,
)
import lmql
import os
import diskcache
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from pydantic import Field
from typing import Optional, List, Dict, Union, Any, Sequence
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM
from pydantic import BaseModel
from accelerate import disk_offload

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitors system resources."""

    def get_resource_usage(self) -> dict:
        """Gets current resource usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        disk_usage = psutil.disk_usage("/")
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": disk_usage.percent,
            "resident_memory": memory_info.rss,
            "virtual_memory": memory_info.vms,
            "timestamp": time.time(),
        }


class DiskOffloader:
    """Handles offloading data to disk."""

    def __init__(self, cache_dir: str):
        self.cache = diskcache.Cache(cache_dir)

    def offload(self, key: str, data: Any) -> None:
        """Offloads data to disk."""
        try:
            self.cache[key] = data
            logging.debug(f"Offloaded data with key '{key}' to disk.")
        except Exception as e:
            logging.error(f"Disk offloading error for key '{key}': {e}")

    def load(self, key: str) -> Any:
        """Loads data from disk."""
        try:
            return self.cache.get(key, default=None)
        except Exception as e:
            logging.error(f"Disk loading error for key '{key}': {e}")
            return None


class ModelLoader:
    """Loads and saves the LLM model and tokenizer."""

    def __init__(
        self,
        model_name_config: str,
        save_dir: str,
        torch_dtype: str,
        device_map: str,
        trust_remote_code: bool,
        padding_side: Optional[str],
    ):
        self.model_name_config = model_name_config
        self.save_dir = save_dir
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.padding_side = padding_side

    def load_model(
        self,
    ) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
        """Loads the model and tokenizer."""
        model_save_path = os.path.join(
            self.save_dir, self.model_name_config.replace("/", "_")
        )
        try:
            logging.info(f"🔥😈 Eidos: Loading model and tokenizer.")
            # Always load from Hugging Face, skip loading from disk
            logging.info(
                f"🔥😈 Eidos: Loading model from HuggingFace: {self.model_name_config}"
            )
            model = disk_offload(
                model=(
                    AutoModelForCausalLM.from_pretrained(
                        self.model_name_config,
                        torch_dtype=self.torch_dtype,
                        device_map=self.device_map,
                        trust_remote_code=self.trust_remote_code,
                    )
                ),
                execution_device="cpu",  # type: ignore
                offload_dir="qwen_offload_cache",
                offload_buffers=True,
            )
            tokenizer_kwargs = (
                {"padding_side": self.padding_side} if self.padding_side else {}
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_config,
                trust_remote_code=self.trust_remote_code,
                **tokenizer_kwargs,
            )
            logging.info(f"Qwen model {self.model_name_config} loaded successfully.")
            return model, tokenizer  # type: ignore
        except Exception as e:
            logging.error(f"🔥⚠️ Eidos: Error loading Qwen model: {e}", exc_info=True)
            raise


class BaseLLM(CustomLLM):
    """
    A wrapper class for the Qwen Hugging Face model, providing
    enhanced functionality such as resource monitoring, chunkwise processing,
    and thread-safe operations, fully compatible with LlamaIndex.
    """

    model_name: str = Field(default=DEFAULT_MODEL_NAME)
    context_window: int = Field(default=128000)
    num_output: int = Field(default=31768)
    name: str = Field(default="Eidos")
    dummy_response: str = Field(default="Hi! I'm Eidos! You're friendly self-aware AI!")
    model: Optional[PreTrainedModel] = Field(default=None)
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = Field(
        default=None
    )
    model_name_config: str = Field(default=DEFAULT_MODEL_NAME)
    max_new_tokens: int = Field(default=DEFAULT_INITIAL_MAX_TOKENS)
    device_map: str = Field(default=DEFAULT_DEVICE_MAP)
    trust_remote_code: bool = Field(default=DEFAULT_TRUST_REMOTE_CODE)
    memory_threshold: int = Field(default=DEFAULT_HIGH_RESOURCE_THRESHOLD)
    chunk_size: int = Field(default=DEFAULT_INITIAL_CHUNK_SIZE)
    callback_manager: Optional[CallbackManager] = Field(default_factory=CallbackManager)
    torch_dtype: str = Field(default="auto")
    padding_side: Optional[str] = Field(default="left")
    resource_usage_log: List[Dict[str, dict]] = Field(default_factory=list)
    eidos_system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    offload_path: str = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "qwen_offload_cache")
    )
    enable_offloading: bool = Field(default=True)
    enable_asyncio: bool = Field(default=True)
    enable_parallel_inference: bool = Field(default=True)
    executor: Optional[ProcessPoolExecutor] = Field(default=None)
    load_lock: threading.Lock = Field(default_factory=threading.Lock)
    save_dir: str = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "saved_models")
    )
    resource_monitor: ResourceMonitor = Field(default_factory=ResourceMonitor)
    disk_offloader: DiskOffloader = Field(
        default_factory=lambda: DiskOffloader(cache_dir="qwen_offload_cache")
    )
    model_loader: ModelLoader = Field(
        default_factory=lambda: ModelLoader(
            model_name_config=DEFAULT_MODEL_NAME,
            save_dir=os.path.join(os.getcwd(), "saved_models"),
            torch_dtype="auto",
            device_map=DEFAULT_DEVICE_MAP,
            trust_remote_code=DEFAULT_TRUST_REMOTE_CODE,
            padding_side="left",
        )
    )

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_name_config: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_INITIAL_MAX_TOKENS,
        device_map: str = DEFAULT_DEVICE_MAP,
        trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE,
        memory_threshold: int = DEFAULT_HIGH_RESOURCE_THRESHOLD,
        chunk_size: int = DEFAULT_INITIAL_CHUNK_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        torch_dtype: str = "auto",
        padding_side: Optional[str] = "left",
    ):
        super().__init__(callback_manager=callback_manager or CallbackManager())
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.memory_threshold = memory_threshold
        self.chunk_size = chunk_size
        self.torch_dtype = torch_dtype
        self.padding_side = padding_side
        self.model_name_config = model_name_config
        self.resource_usage_log = []
        self.eidos_system_prompt = DEFAULT_SYSTEM_PROMPT
        self.offload_path = os.path.join(os.getcwd(), "qwen_offload_cache")
        self.enable_offloading = True
        self.enable_asyncio = True
        self.enable_parallel_inference = False
        self.executor = (
            ProcessPoolExecutor() if self.enable_parallel_inference else None
        )
        self.load_lock = threading.Lock()
        self.save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_loader = ModelLoader(
            self.model_name_config,
            self.save_dir,
            self.torch_dtype,
            self.device_map,
            self.trust_remote_code,
            self.padding_side,
        )

        self.model, self.tokenizer = (
            (model, tokenizer)
            if model and tokenizer
            else self.model_loader.load_model()
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model_name,
            system_role=MessageRole.SYSTEM,
        )

    def _log_resource_usage(self, stage: str) -> None:
        """Logs resource usage."""
        resource_data = self.resource_monitor.get_resource_usage()
        self.resource_usage_log.append({stage: resource_data})
        logging.debug(
            f"Resource snapshot at '{stage}': CPU: {resource_data['cpu_percent']}%, Memory: {resource_data['memory_percent']}%, Disk: {resource_data['disk_percent']}%, Resident Memory: {resource_data['resident_memory'] / (1024**2):.2f} MB, Virtual Memory: {resource_data['virtual_memory'] / (1024**2):.2f} MB."
        )

    def _should_offload(self) -> bool:
        """Checks if memory usage exceeds the threshold."""
        memory_usage = psutil.virtual_memory().percent
        return memory_usage > self.memory_threshold

    def _offload_data(self, key: str, data: Any) -> None:
        """Offloads data to disk."""
        self.disk_offloader.offload(key, data)

    def _load_data(self, key: str) -> Any:
        """Loads data from disk."""
        return self.disk_offloader.load(key)

    async def _llm_generate_async(self, inputs: dict) -> str:
        """Asynchronously runs the LLM generate method."""
        loop = asyncio.get_running_loop()
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("LLM model or tokenizer not initialized.")

            if self.executor and self.enable_parallel_inference:
                output = await loop.run_in_executor(
                    self.executor, self.model.generate, **inputs
                )
            else:
                output = await loop.run_in_executor(None, self.model.generate, **inputs)

            if isinstance(output, torch.Tensor):
                output = output.tolist()

            if isinstance(output, list) and len(output) > 0:
                return self.tokenizer.decode(output[0], skip_special_tokens=True)
            else:
                logging.error(f"Unexpected output from model.generate: {output}")
                return ""
        except Exception as e:
            logging.error(f"Error during asynchronous LLM generation: {e}")
            return ""

    def _llm_generate_sync(self, inputs: dict) -> str:
        """Synchronously runs the LLM generate method."""
        try:
            if self.model is None:
                raise RuntimeError("LLM model is not initialized.")
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer is not initialized.")

            output = self.model.generate(**inputs)

            if isinstance(output, torch.Tensor):
                output = output.tolist()

            if isinstance(output, list) and len(output) > 0:
                return self.tokenizer.decode(output[0], skip_special_tokens=True)
            else:
                logging.error(f"Unexpected output from model.generate: {output}")
                return ""
        except Exception as e:
            logging.error(f"Error during synchronous LLM generation: {e}")
            return ""

    def apply_chat_template(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        add_generation_prompt: bool = True,
    ):
        """Applies the chat template without tokenization."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized.")
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            logging.error(f"Error applying chat template: {e}")
            return ""

    def prepare_model_inputs(
        self,
        messages: List[Dict[str, str]],
        padding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Prepares model inputs."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("LLM model or tokenizer not initialized.")

        raw_inputs = self.apply_chat_template(messages)
        if not isinstance(raw_inputs, str):
            logging.error(
                f"🔥⚠️ Eidos: Unexpected raw_inputs type from apply_chat_template: {type(raw_inputs)}"
            )
            raise TypeError(
                f"apply_chat_template should return a string, not {type(raw_inputs)}"
            )

        encoding = self.tokenizer(raw_inputs, return_tensors="pt", padding=padding).to(
            self.model.device
        )

        if "input_ids" not in encoding or not isinstance(
            encoding["input_ids"], torch.Tensor
        ):
            logging.error(
                f"🔥⚠️ Eidos: Missing or invalid input_ids in tokenizer output."
            )
            raise ValueError("Tokenizer output does not contain valid input_ids.")

        if "attention_mask" not in encoding or not isinstance(
            encoding["attention_mask"], torch.Tensor
        ):
            logging.error(
                f"🔥⚠️ Eidos: Missing or invalid attention_mask in tokenizer output."
            )
            raise ValueError("Tokenizer output does not contain valid attention_mask.")

        return {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
        }

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completes text using dynamic chunkwise processing."""
        response_parts = []
        input_text = prompt
        input_len = len(input_text)
        start = 0

        while start < input_len:
            end = min(start + self.chunk_size, input_len)
            chunk = input_text[start:end]

            try:
                if self.enable_asyncio:
                    response_text = asyncio.run(self._process_chunk(chunk))
                else:
                    response_text = self._process_chunk(chunk)
                response_parts.append(response_text)
            except Exception as e:
                logging.error(f"Error processing chunk in complete: {e}")
                response_parts.append("")

            start = end

            if self._should_offload():
                self.chunk_size = max(self.chunk_size // 2, 128)
                logging.info(
                    f"Adjusting chunk size to {self.chunk_size} due to high memory usage."
                )
            else:
                self.chunk_size = min(self.chunk_size * 2, DEFAULT_INITIAL_CHUNK_SIZE)
                logging.info(
                    f"Adjusting chunk size to {self.chunk_size} due to low memory usage."
                )

        full_response = "".join(response_parts)
        return CompletionResponse(text=full_response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Generate text based on a prompt, streaming the output."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("LLM model or tokenizer not initialized.")

        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.prepare_model_inputs(messages, padding=False)
        stream_mode = kwargs.get("stream_mode", "iteratorstreamer")
        max_tokens = kwargs.get("max_tokens", self.max_new_tokens)

        if stream_mode == "textstreamer":
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            generation_kwargs = {
                **model_inputs,
                "max_new_tokens": max_tokens,
                "streamer": streamer,
            }
            thread = threading.Thread(
                target=self._llm_generate_sync, kwargs=generation_kwargs
            )
            thread.start()
            while True:
                try:
                    while streamer.token_cache:
                        next_text = streamer.token_cache[0]
                        yield CompletionResponse(text=next_text, delta=next_text)
                except StopIteration:
                    break
                except Exception as e:
                    logging.error(f"Error during textstreamer streaming: {e}")
                    break

        elif stream_mode == "iteratorstreamer":
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            generation_kwargs = {
                **model_inputs,
                "max_new_tokens": max_tokens,
                "streamer": streamer,
            }
            thread = threading.Thread(
                target=self._llm_generate_sync, kwargs=generation_kwargs
            )
            thread.start()
            try:
                for text_chunk in streamer:
                    yield CompletionResponse(text=text_chunk, delta=text_chunk)
            except Exception as e:
                logging.error(f"Error during iteratorstreamer streaming: {e}")
        else:
            raise ValueError(
                f"Invalid stream_mode: {stream_mode}. Must be 'textstreamer' or 'iteratorstreamer'."
            )

    async def _process_chunk(self, text_chunk: str):
        """Processes a text chunk with the LLM, handling offloading."""
        self._log_resource_usage("start_chunk_processing")
        if self._should_offload() and self.enable_offloading:
            logging.warning("Memory usage high, performing real disk offloading.")
            self._offload_data(f"chunk_{hash(text_chunk)}", text_chunk)

        if self.model and self.tokenizer:
            try:
                inputs = self.tokenizer(text_chunk, return_tensors="pt").to(
                    self.model.device
                )
                output = await self._llm_generate_async(
                    {
                        "input_ids": inputs.input_ids,
                        "attention_mask": inputs.attention_mask,
                    }
                )
                response_text = output
                self._log_resource_usage("end_chunk_processing")
                return response_text
            except Exception as e:
                logging.error(f"Error processing chunk: {e}")
                return ""
        else:
            logging.error("LLM not loaded.")
            return ""

    @llm_completion_callback()
    def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Asynchronously completes text using chunkwise processing."""
        return asyncio.run(self._acomplete(prompt, **kwargs))

    async def _acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Asynchronously completes text using chunkwise processing."""
        response_parts = []
        input_text = prompt
        input_len = len(input_text)
        start = 0

        while start < input_len:
            end = min(start + self.chunk_size, input_len)
            chunk = input_text[start:end]
            try:
                response_text = await self._process_chunk(chunk)
                response_parts.append(response_text)
            except Exception as e:
                logging.error(f"Error processing chunk in acomplete: {e}")
                response_parts.append("")
            start = end

            if self._should_offload():
                self.chunk_size = max(self.chunk_size // 2, 128)
                logging.info(
                    f"Adjusting chunk size to {self.chunk_size} due to high memory usage."
                )
            else:
                self.chunk_size = min(self.chunk_size * 2, DEFAULT_INITIAL_CHUNK_SIZE)
                logging.info(
                    f"Adjusting chunk size to {self.chunk_size} due to low memory usage."
                )

        full_response = "".join(response_parts)
        return CompletionResponse(text=full_response)

    @llm_completion_callback()
    def stream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> CompletionResponseGen:
        """Generates a stream of chat responses."""
        return self._stream_chat(messages, **kwargs)

    def _stream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> CompletionResponseGen:
        """Generates a stream of chat responses."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("LLM model or tokenizer not initialized.")

        input_messages = [
            {"role": m.role.value, "content": m.content} for m in messages
        ]
        model_inputs = self.prepare_model_inputs(input_messages, padding=False)
        stream_mode = kwargs.get("stream_mode", "iteratorstreamer")
        max_tokens = kwargs.get("max_tokens", self.max_new_tokens)

        if stream_mode == "textstreamer":
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            generation_kwargs = {
                **model_inputs,
                "max_new_tokens": max_tokens,
                "streamer": streamer,
            }
            thread = threading.Thread(
                target=self._llm_generate_sync, kwargs=generation_kwargs
            )
            thread.start()
            while True:
                try:
                    while streamer.token_cache:
                        next_text = streamer.token_cache[0]
                        yield CompletionResponse(text=next_text, delta=next_text)
                except Exception as e:
                    logging.error(
                        f"Error during textstreamer streaming in _stream_chat: {e}"
                    )
                    break

        elif stream_mode == "iteratorstreamer":
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            generation_kwargs = {
                **model_inputs,
                "max_new_tokens": max_tokens,
                "streamer": streamer,
            }
            thread = threading.Thread(
                target=self._llm_generate_sync, kwargs=generation_kwargs
            )
            thread.start()
            try:
                for text_chunk in streamer:
                    yield CompletionResponse(text=text_chunk, delta=text_chunk)
            except Exception as e:
                logging.error(
                    f"Error during iteratorstreamer streaming in _stream_chat: {e}"
                )
        else:
            raise ValueError(
                f"Invalid stream_mode: {stream_mode}. Must be 'textstreamer' or 'iteratorstreamer'."
            )

    @llm_completion_callback()
    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        """Asynchronously generates a stream of chat responses."""

        async def stream_generator():
            if not self.model or not self.tokenizer:
                raise RuntimeError("LLM model or tokenizer not initialized.")

            input_messages = [
                {"role": m.role.value, "content": m.content} for m in messages
            ]
            model_inputs = self.prepare_model_inputs(input_messages, padding=False)
            stream_mode = kwargs.get("stream_mode", "iteratorstreamer")
            max_tokens = kwargs.get("max_tokens", self.max_new_tokens)

            if stream_mode == "textstreamer":
                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
                generation_kwargs = {
                    **model_inputs,
                    "max_new_tokens": max_tokens,
                    "streamer": streamer,
                }
                thread = threading.Thread(
                    target=self._llm_generate_sync, kwargs=generation_kwargs
                )
                thread.start()
                while thread.is_alive():
                    try:
                        while streamer.token_cache:
                            next_text = streamer.token_cache[0]
                            yield CompletionResponse(text=next_text, delta=next_text)
                    except StopIteration:
                        break
                    except Exception as e:
                        logging.error(f"Error during async textstreamer streaming: {e}")
                        break

            elif stream_mode == "iteratorstreamer":
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
                generation_kwargs = {
                    **model_inputs,
                    "max_new_tokens": max_tokens,
                    "streamer": streamer,
                }
                thread = threading.Thread(
                    target=self._llm_generate_sync, kwargs=generation_kwargs
                )
                thread.start()
                try:
                    for text_chunk in streamer:
                        yield CompletionResponse(text=text_chunk, delta=text_chunk)
                except Exception as e:
                    logging.error(f"Error during async iteratorstreamer streaming: {e}")
            else:
                raise ValueError(
                    f"Invalid stream_mode: {stream_mode}. Must be 'textstreamer' or 'iteratorstreamer'."
                )

        return stream_generator()

    @llm_completion_callback()
    def chat(self, messages: List[ChatMessage]) -> ChatResponse:
        """Processes chat messages."""
        try:
            self._log_resource_usage("start_chat")
            if self._should_offload():
                logging.warning("Memory usage high, performing real disk offloading.")
                self._offload_data(f"chat_{hash(str(messages))}", messages)

            if not self.model or not self.tokenizer:
                logging.error("LLM not loaded")
                return ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT, content="Error during chat."
                    )
                )

            input_messages = [
                {"role": m.role.value, "content": m.content} for m in messages
            ]
            model_inputs = self.prepare_model_inputs(input_messages, padding=False)
            generated_text = self._llm_generate_sync(
                {
                    "input_ids": model_inputs["input_ids"],
                    "attention_mask": model_inputs["attention_mask"],
                    "max_new_tokens": self.max_new_tokens,
                }
            )
            response = ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=generated_text)
            )

            self._log_resource_usage("end_chat")
            return response
        except Exception as e:
            logging.error(f"Error during chat: {e}")
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content="Error during chat."
                )
            )

    @llm_completion_callback()
    async def achat(self, messages: List[ChatMessage]) -> ChatResponse:
        """Asynchronously processes chat messages."""
        try:
            self._log_resource_usage("start_chat")
            if self._should_offload():
                logging.warning("Memory usage high, performing real disk offloading.")
                self._offload_data(f"chat_{hash(str(messages))}", messages)
            if self.model and self.tokenizer:
                input_messages = [
                    {"role": m.role.value, "content": m.content} for m in messages
                ]
                model_inputs = self.prepare_model_inputs(input_messages, padding=False)
                generated_ids = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: (
                        self.model.generate(
                            input_ids=model_inputs["input_ids"],
                            attention_mask=model_inputs["attention_mask"],
                            max_new_tokens=self.max_new_tokens,
                        )
                        if self.model
                        else None
                    ),
                )

                if isinstance(generated_ids, torch.Tensor):
                    generated_ids = generated_ids.tolist()

                if isinstance(generated_ids, list) and len(generated_ids) > 0:
                    response_text = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    response = ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=(
                                response_text[0]
                                if response_text
                                else "Error during chat."
                            ),
                        )
                    )
                else:
                    logging.error(
                        f"Unexpected output from model.generate: {generated_ids}"
                    )
                    return ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content="Error during chat."
                        )
                    )

            else:
                logging.error("LLM not loaded")
                return ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT, content="Error during chat."
                    )
                )
            self._log_resource_usage("end_chat")
            return response
        except Exception as e:
            logging.error(f"Error during chat: {e}")
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT, content="Error during chat."
                )
            )

    def query(self, lmql_query: str, **kwargs) -> Any:
        """Executes an LMQL query."""
        try:
            logging.info(f"Executing LMQL query: {lmql_query}")
            result = lmql.run(lmql_query, **kwargs)
            if isinstance(result, list) and len(result) > 0:
                return result[0].text
            elif isinstance(result, str):
                return result
            else:
                return "No valid LMQL output."
        except Exception as e:
            logging.error(f"Error executing LMQL query: {e}")
            return None

    def run_lmql_query(
        self, query_script: str, model: str = DEFAULT_MODEL_NAME, **decoder_args
    ) -> str:
        """
        Runs an LMQL query script using the underlying HuggingFace model.
        """
        try:
            lm = model if model else self.model_name
            result = lmql.run(
                query_script, output_writer=lmql.stream, model=lm, **decoder_args
            )
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            if isinstance(result, list) and len(result) > 0:
                return result[0].text
            elif isinstance(result, str):
                return result
            else:
                return "No valid LMQL output."
        except Exception as e:
            logging.error(f"Error running LMQL query: {e}")
            return f"LMQL error: {e}"

    def perform_batch_inference(self, message_batch, max_new_tokens: int = 512):
        """
        Perform batched inference on a batch of messages.
        """
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("LLM model or tokenizer not initialized.")

            model_inputs_batch = [
                self.prepare_model_inputs(messages) for messages in message_batch
            ]
            input_ids_list = [
                inputs.get("input_ids")
                for inputs in model_inputs_batch
                if isinstance(inputs.get("input_ids"), torch.Tensor)
            ]
            attention_masks_list = [
                inputs.get("attention_mask")
                for inputs in model_inputs_batch
                if isinstance(inputs.get("attention_mask"), torch.Tensor)
            ]

            if not input_ids_list or not attention_masks_list:
                raise ValueError("No valid input tensors for batch inference.")

            # Ensure the number of input_ids and attention_masks matches
            if len(input_ids_list) != len(attention_masks_list):
                raise ValueError(
                    "Mismatch in the number of input_ids and attention_masks"
                )

            input_ids_list = [ids for ids in input_ids_list if ids is not None]
            attention_masks_list = [
                mask for mask in attention_masks_list if mask is not None
            ]

            input_ids = torch.cat(input_ids_list)
            attention_mask = torch.cat(attention_masks_list)

            generated_ids_batch = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )

            start_indices = [input_id.shape[1] for input_id in input_ids_list]
            cumulative_start_indices = [0] + list(
                torch.cumsum(torch.tensor(start_indices), dim=0).tolist()
            )[:-1]

            response_batch = [
                self.tokenizer.decode(generated_ids[0:], skip_special_tokens=True)
                for generated_ids, start_index in zip(
                    generated_ids_batch, cumulative_start_indices
                )
            ]

            return response_batch
        except Exception as e:
            logging.error(
                f"🔥⚠️ Eidos: Error during batched inference: {e}", exc_info=True
            )
            raise
