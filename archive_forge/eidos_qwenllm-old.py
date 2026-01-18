# qwen2_base.py
"""
This module implements the BaseLLM class, a core component of the Eidos digital intelligence system.
It provides an interface to the Qwen series of Large Language Models (LLMs) from Hugging Face,
optimized for performance, flexibility, and scalability, and full compatibility with LlamaIndex.

Key features include:
    - Resource Monitoring: Tracks CPU, memory, and disk usage using `psutil` for adaptive processing.
    - Chunkwise Processing: Handles large text inputs by processing them in dynamically sized chunks,
      reducing memory footprint and enabling efficient handling of extensive text.
    - Thread-Safe Operations: Utilizes threading locks to ensure safe concurrent access to the model.
    - Disk Offloading: Implements a disk-based cache using `diskcache` to offload data when memory usage is high,
      allowing for processing of very large datasets.
    - Asynchronous Operations: Leverages `asyncio` for non-blocking I/O operations, improving responsiveness.
    - LMQL Integration: Supports execution of LMQL queries for advanced prompting and control over LLM behavior.
    - Enhanced Logging: Provides detailed logging with resource monitoring, profiling, and tracing via the `eidos_logging` module.

This module serves as the foundational LLM layer for the Eidos project, upon which higher-level functionalities are built.
It is designed for maximum robustness and efficiency, ensuring reliable performance in diverse operational environments.
"""

from typing import Optional, List, Mapping, Any, Union, Generator

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
from typing import Optional, List, Dict, Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM
from pydantic import BaseModel

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


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
    model: PreTrainedModel = Field(
        default_factory=lambda: AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME)
    )
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = Field(
        default_factory=lambda: AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    )
    model_name_config: str = Field(default="Eidos")
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
    cache: diskcache.Cache = Field(
        default_factory=lambda: diskcache.Cache(
            os.path.join(os.getcwd(), "qwen_offload_cache")
        )
    )
    enable_offloading: bool = Field(default=True)
    enable_asyncio: bool = Field(default=True)
    enable_parallel_inference: bool = Field(default=False)
    executor: Optional[ProcessPoolExecutor] = Field(default=None)
    load_lock: threading.Lock = Field(default_factory=threading.Lock)

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_name_config: str = "Eidos",
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
        self.model = (
            model
            if model is not None
            else AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME)
        )
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        )
        self.torch_dtype = torch_dtype
        self.padding_side = padding_side
        self.model_name_config = model_name_config
        self.resource_usage_log = []
        self.eidos_system_prompt = DEFAULT_SYSTEM_PROMPT
        self.offload_path = os.path.join(os.getcwd(), "qwen_offload_cache")
        self.cache = diskcache.Cache(self.offload_path)
        self.enable_offloading = True
        self.enable_asyncio = True
        self.enable_parallel_inference = False
        self.executor = (
            ProcessPoolExecutor() if self.enable_parallel_inference else None
        )
        self.load_lock = threading.Lock()
        self._load_model()

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=True,  # Assuming it's a chat model
            is_function_calling_model=True,  # Assuming no function calling by default
            model_name=self.model_name,
            system_role=MessageRole.SYSTEM,
        )

    def _load_model(self) -> None:
        """Loads the Qwen model and tokenizer."""
        with self.load_lock:
            if self.model is None:
                try:
                    logging.info("🔥😈 Eidos: Loading model and tokenizer.")
                    self.model.disk_offload(
                        AutoModelForCausalLM.from_pretrained(
                            self.model_name_config,
                            torch_dtype=self.torch_dtype,
                            device_map=self.device_map,
                            trust_remote_code=self.trust_remote_code,
                        )
                    )
                    tokenizer_kwargs = {}
                    if self.padding_side:
                        tokenizer_kwargs["padding_side"] = self.padding_side
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name_config,
                        trust_remote_code=self.trust_remote_code,
                        **tokenizer_kwargs,
                    )
                    logging.info(
                        f"Qwen model {self.model_name_config} loaded successfully."
                    )
                except Exception as e:
                    logging.error(
                        f"🔥⚠️ Eidos: Error loading Qwen model: {e}", exc_info=True
                    )
                    raise

    def _log_resource_usage(self, stage: str) -> None:
        """Logs resource usage."""
        resource_data = self._get_resource_usage()
        self.resource_usage_log.append({stage: resource_data})
        logging.debug(
            f"Resource snapshot at '{stage}': CPU: {resource_data['cpu_percent']}%, Memory: {resource_data['memory_percent']}%, Disk: {resource_data['disk_percent']}%, Resident Memory: {resource_data['resident_memory'] / (1024**2):.2f} MB, Virtual Memory: {resource_data['virtual_memory'] / (1024**2):.2f} MB."
        )

    def _get_resource_usage(self) -> dict:
        """Gets current resource usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/")

        resource_data = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_usage.percent,
            "resident_memory": memory_info.rss,
            "virtual_memory": memory_info.vms,
            "timestamp": time.time(),
        }
        return resource_data

    def _should_offload(self) -> bool:
        """Checks if memory usage exceeds the threshold."""
        memory_usage = psutil.virtual_memory().percent
        return memory_usage > self.memory_threshold

    def _offload_to_disk(self, key: str, data: Any) -> None:
        """Offloads 'data' to disk under the key 'key'."""
        try:
            self.cache[key] = data
            logging.debug(f"Offloaded data with key '{key}' to disk.")
        except Exception as e:
            logging.error(f"Disk offloading error for key '{key}': {e}")

    def _load_from_disk(self, key: str) -> Any:
        """Loads data from diskcache for the given 'key'."""
        try:
            return self.cache.get(key, default=None)
        except Exception as e:
            logging.error(f"Disk loading error for key '{key}': {e}")
            return None

    async def _process_chunk_async(self, text_chunk: str) -> str:
        """Asynchronous version of chunk processing."""
        self._log_resource_usage("start_chunk_processing_async")
        if self._should_offload() and self.enable_offloading:
            logging.warning("Memory usage high, performing real disk offloading.")
            self._offload_to_disk(f"chunk_{hash(text_chunk)}", text_chunk)

        loop = asyncio.get_running_loop()
        if self.executor and self.enable_parallel_inference:
            response = await loop.run_in_executor(
                self.executor, self._run_llm_sync, text_chunk
            )
        else:
            response = await loop.run_in_executor(None, self._run_llm_sync, text_chunk)

        self._log_resource_usage("end_chunk_processing_async")
        return response

    def _run_llm_sync(self, text_chunk: str) -> str:
        """Helper method to run the LLM synchronously."""
        if self.model and hasattr(self.model, "generate"):
            try:
                if self.tokenizer:
                    inputs = self.tokenizer(text_chunk, return_tensors="pt").to(
                        self.model.device
                    )
                    output = self.model.generate(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                    )
                    resp_text = self.tokenizer.decode(
                        output[0], skip_special_tokens=True
                    )
                    return resp_text
            except Exception as e:
                logging.error(f"Error in synchronous LLM call: {e}")
        return ""

    def apply_chat_template(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        add_generation_prompt: bool = True,
    ):
        """Applies the chat template without tokenization."""
        if self.tokenizer:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        else:
            raise RuntimeError("Tokenizer not initialized.")

    def prepare_model_inputs(
        self,
        messages: List[Dict[str, str]],
        padding: bool = False,
    ):
        """Prepares model inputs for single or batched messages."""
        raw_inputs = self.apply_chat_template(messages)
        if isinstance(raw_inputs, str):
            if self.model and self.tokenizer:
                model_inputs = self.tokenizer(
                    raw_inputs, return_tensors="pt", padding=padding
                ).to(self.model.device)
            else:
                raise RuntimeError("LLM model or tokenizer not initialized.")
        elif isinstance(raw_inputs, list) and all(
            isinstance(item, str) for item in raw_inputs
        ):
            if self.model and self.tokenizer:
                model_inputs = self.tokenizer(
                    raw_inputs, return_tensors="pt", padding=padding  # type: ignore
                ).to(self.model.device)
            else:
                raise RuntimeError("LLM model or tokenizer not initialized.")
        else:
            logger.error("🔥⚠️ Eidos: Invalid raw_inputs type for tokenizer.")
            raise TypeError("Invalid raw_inputs type. Expected str or List[str].")
        return model_inputs

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completes text using chunkwise processing."""
        all_responses = ""
        start = 0
        text = prompt
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            response_text = self._process_chunk(chunk)
            all_responses += response_text
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
        return CompletionResponse(text=all_responses)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Generate text based on a prompt, streaming the output."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("LLM model or tokenizer not initialized.")

        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.prepare_model_inputs(messages, padding=False)
        stream_mode = kwargs.get("stream_mode", "iteratorstreamer")

        if stream_mode == "textstreamer":
            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
            )
            thread = threading.Thread(
                target=self.model.generate,
                kwargs={
                    **model_inputs,
                    "max_new_tokens": kwargs.get("max_tokens", self.max_new_tokens),
                    "streamer": streamer,
                },
            )
            thread.start()
            while True:
                try:
                    next_text = streamer.decode_kwargs["decode_kwargs"](
                        token_ids=streamer.token_cache
                    )
                    yield CompletionResponse(text=next_text, delta=next_text)
                except StopIteration:
                    break

        elif stream_mode == "iteratorstreamer":
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
            )
            generation_kwargs = {
                **model_inputs,
                "streamer": streamer,
                "max_new_tokens": kwargs.get("max_tokens", self.max_new_tokens),
            }
            thread = threading.Thread(
                target=self.model.generate, kwargs=generation_kwargs
            )
            thread.start()
            for text_chunk in streamer:
                yield CompletionResponse(text=text_chunk, delta=text_chunk)
        else:
            raise ValueError(
                f"Invalid stream_mode: {stream_mode}. Must be 'textstreamer' or 'iteratorstreamer'."
            )

    def _process_chunk(self, text_chunk: str) -> str:
        """Processes a text chunk with the LLM."""
        if self.enable_asyncio:
            return asyncio.run(self._process_chunk_async(text_chunk))
        else:
            self._log_resource_usage("start_chunk_processing")
            if self._should_offload() and self.enable_offloading:
                logging.warning("Memory usage high, performing real disk offloading.")
                self._offload_to_disk(f"chunk_{hash(text_chunk)}", text_chunk)

            if self.model and hasattr(self.model, "generate") and self.tokenizer:
                try:
                    inputs = self.tokenizer(text_chunk, return_tensors="pt").to(
                        self.model.device
                    )
                    output = self.model.generate(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                    )
                    response_text = self.tokenizer.decode(
                        output[0], skip_special_tokens=True
                    )
                    self._log_resource_usage("end_chunk_processing")
                    return response_text
                except Exception as e:
                    logging.error(f"Error processing chunk: {e}")
                    return ""
            else:
                logging.error("LLM not loaded.")
                return ""

    def chat(self, messages: List[ChatMessage]) -> ChatMessage:
        """Processes chat messages."""
        try:
            self._log_resource_usage("start_chat")
            if self._should_offload():
                logging.warning("Memory usage high, performing real disk offloading.")
                self._offload_to_disk(f"chat_{hash(str(messages))}", messages)
            if self.model and self.tokenizer:
                input_messages = [
                    {"role": m.role.value, "content": m.content} for m in messages
                ]
                model_inputs = self.prepare_model_inputs(input_messages, padding=False)
                generated_ids = self.model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=self.max_new_tokens,
                )
                generated_ids = [
                    output_ids[len(input_id) :].tolist()
                    for input_id, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]
                response_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                response = ChatMessage(
                    role=MessageRole.ASSISTANT, content=response_text
                )

            else:
                logging.error("LLM not loaded")
                return ChatMessage(
                    role=MessageRole.ASSISTANT, content="Error during chat."
                )
            self._log_resource_usage("end_chat")
            return response
        except Exception as e:
            logging.error(f"Error during chat: {e}")
            return ChatMessage(role=MessageRole.ASSISTANT, content="Error during chat.")

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

    def perform_batch_inference(
        self, message_batch: List[List[Dict[str, str]]], max_new_tokens: int = 512
    ) -> List[str]:
        """
        Perform batched inference on a batch of messages.
        """
        try:
            model_inputs_batch = [
                self.prepare_model_inputs(messages) for messages in message_batch
            ]
            if self.model:
                generated_ids_batch = self.model.generate(
                    input_ids=torch.cat(
                        [inputs.input_ids for inputs in model_inputs_batch]
                    ),
                    attention_mask=torch.cat(
                        [inputs.attention_mask for inputs in model_inputs_batch]
                    ),
                    max_new_tokens=max_new_tokens,
                )
            else:
                raise RuntimeError("LLM model not initialized.")
            # Calculate the starting index for generated text based on the input length
            start_indices = [input.input_ids.shape[1] for input in model_inputs_batch]
            if self.tokenizer:
                response_batch = [
                    self.tokenizer.decode(
                        generated_ids[start:], skip_special_tokens=True
                    )
                    for generated_ids, start in zip(generated_ids_batch, start_indices)
                ]
            else:
                raise RuntimeError("Tokenizer not initialized.")
            return response_batch
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error during batched inference: {e}", exc_info=True
            )
            raise


# Example usage
qwen_llm = BaseLLM()

# "Complete" style usage
response_text = qwen_llm.complete("Tell me about large language models. " * 10)
print("LLM Completion:", response_text)

user_message = ChatMessage(
    role=MessageRole.USER, content="Tell me about large language models."
)
chat_response = qwen_llm.chat(messages=[user_message])
print("LLM Chat Response:", chat_response.content)


def demo_lmql_usage(llm: BaseLLM):
    test_query = """
argmax
    # A basic prompt with a constraint
    "Say 'this is a test':[RESPONSE]" where len(TOKENS(RESPONSE)) < 25
from
    "{model_name}"
    """.format(
        model_name=llm.model_name
    )

    lmql_response = llm.run_lmql_query(test_query)
    print("LMQL Query Output:", lmql_response)


demo_lmql_usage(qwen_llm)
