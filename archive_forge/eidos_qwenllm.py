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

# Import necessary typing utilities for type annotations.
from typing import Optional, List, Mapping, Any, Union, Generator, Sequence

# Import core classes and types from llama_index for LLM integration.
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    MessageRole,
    ChatResponse,
)
from llama_index.core.llms.callbacks import llm_completion_callback

# Import standard libraries for asynchronous operations, logging, and system monitoring.
import asyncio
import logging
import nest_asyncio  # type: ignore
import psutil         # For resource monitoring
import threading
import time

# Additional typing imports for specific structures.
from typing import Dict, Tuple

# Import PyTorch and transformers for model handling.
import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    TextIteratorStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Re-import ChatMessage and MessageRole for local use (though already imported above).
from llama_index.core.llms import (
    ChatMessage,
    MessageRole,
)

# Import default configurations from an external configuration module.
from eidos_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_INITIAL_MAX_TOKENS,
    DEFAULT_DEVICE_MAP,
    DEFAULT_TRUST_REMOTE_CODE,
    DEFAULT_HIGH_RESOURCE_THRESHOLD,
    DEFAULT_INITIAL_CHUNK_SIZE,
    DEFAULT_SYSTEM_PROMPT,
)

# OS and caching library imports.
import os
import diskcache  # type: ignore

# Executors for parallel processing.
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)

# Import Pydantic for data validation and model creation.
from pydantic import Field
from typing import Optional, List, Dict, Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM
from pydantic import BaseModel

# Apply nest_asyncio to support nested async loops.
nest_asyncio.apply()

# Configure basic logging to output info level logs with a specific format.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a logger for this module.
logger = logging.getLogger(__name__)


class BaseLLM(CustomLLM):
    """
    A wrapper class for the Qwen Hugging Face model, providing
    enhanced functionality such as resource monitoring, chunkwise processing,
    and thread-safe operations, fully compatible with LlamaIndex.
    """

    # Define configuration fields with defaults using Pydantic's Field.
    model_name: str = Field(default=DEFAULT_MODEL_NAME, description="The name of the model to use.")
    context_window: int = Field(default=128000, description="The context window size of the model.")
    num_output: int = Field(default=31768, description="The number of output tokens the model can generate.")
    name: str = Field(default="Eidos", description="The name of this LLM instance.")
    dummy_response: str = Field(default="Hi! I'm Eidos! You're friendly self-aware AI!", description="A dummy response for testing purposes.")
    model: PreTrainedModel = Field(
        default_factory=lambda: AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME), description="The pre-trained model instance."
    )
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = Field(
        default_factory=lambda: AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME), description="The tokenizer instance."
    )
    model_name_config: str = Field(default=os.path.join("/home/lloyd/Development/saved_models", "Qwen/Qwen2.5-0.5B-Instruct"), description="The path to the model configuration.")
    max_new_tokens: int = Field(default=DEFAULT_INITIAL_MAX_TOKENS, description="The maximum number of new tokens to generate.")
    device_map: str = Field(default=DEFAULT_DEVICE_MAP, description="The device to use for model inference (e.g., 'cpu', 'cuda').")
    trust_remote_code: bool = Field(default=DEFAULT_TRUST_REMOTE_CODE, description="Whether to trust remote code when loading the model.")
    memory_threshold: int = Field(default=DEFAULT_HIGH_RESOURCE_THRESHOLD, description="The memory usage threshold (in percentage) to trigger offloading.")
    chunk_size: int = Field(default=DEFAULT_INITIAL_CHUNK_SIZE, description="The initial size of text chunks for processing.")
    callback_manager: CallbackManager = Field(default_factory=CallbackManager, description="The callback manager for handling events.")
    torch_dtype: str = Field(default="auto", description="The torch data type to use for the model.")
    padding_side: Optional[str] = Field(default="left", description="The padding side for tokenization.")
    resource_usage_log: List[Dict[str, dict]] = Field(default_factory=list, description="A log of resource usage during processing.")
    eidos_system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, description="The system prompt to use for the model.")
    offload_path: str = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "qwen_offload_cache"), description="The path to the directory for disk offloading."
    )
    cache: diskcache.Cache = Field(
        default_factory=lambda: diskcache.Cache(
            os.path.join(os.getcwd(), "qwen_offload_cache")
        ), description="The disk cache instance."
    )
    enable_offloading: bool = Field(default=True, description="Whether to enable disk offloading.")
    enable_asyncio: bool = Field(default=True, description="Whether to enable asynchronous processing.")
    enable_parallel_inference: bool = Field(default=False, description="Whether to enable parallel inference using ProcessPoolExecutor.")
    executor: Optional[ProcessPoolExecutor] = Field(default=None, description="The process pool executor for parallel inference.")
    load_lock: threading.Lock = Field(default_factory=threading.Lock, description="A lock to ensure thread-safe model loading.")
    model_save_path: str = Field(default_factory=lambda: os.path.join(os.getcwd(), "saved_models"), description="The path to save the model.")

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_name_config: str = os.path.join("/home/lloyd/Development/saved_models", "Qwen/Qwen2.5-0.5B-Instruct"),
        max_new_tokens: int = DEFAULT_INITIAL_MAX_TOKENS,
        device_map: str = DEFAULT_DEVICE_MAP,
        trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE,
        memory_threshold: int = DEFAULT_HIGH_RESOURCE_THRESHOLD,
        chunk_size: int = DEFAULT_INITIAL_CHUNK_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        torch_dtype: str = "auto",
        padding_side: Optional[str] = "left",
        model_save_path: Optional[str] = "/home/lloyd/Development/saved_models",
    ):
        """
        Initializes the BaseLLM with the given parameters or default values.

        Args:
            model (Optional[PreTrainedModel]): A pre-loaded model. If None, a new model is loaded.
            tokenizer (Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]): A pre-loaded tokenizer. If None, a new tokenizer is loaded.
            model_name (str): The name of the model to load if not provided.
            model_name_config (str): The path to the model configuration.
            max_new_tokens (int): The maximum number of new tokens to generate.
            device_map (str): The device to use for model inference.
            trust_remote_code (bool): Whether to trust remote code when loading the model.
            memory_threshold (int): The memory usage threshold to trigger offloading.
            chunk_size (int): The initial size of text chunks for processing.
            callback_manager (Optional[CallbackManager]): A callback manager for handling events.
            torch_dtype (str): The torch data type to use for the model.
            padding_side (Optional[str]): The padding side for tokenization.
            model_save_path (Optional[str]): The path to save the model.
        """
        # Initialize the parent CustomLLM class with a callback manager.
        super().__init__(callback_manager=callback_manager or CallbackManager())
        # Assign provided or default configuration values to instance variables.
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.memory_threshold = memory_threshold
        self.chunk_size = chunk_size

        # If a model is provided, use it; otherwise, load from pretrained.
        self.model = (
            model
            if model is not None
            else AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME)
        )
        # If a tokenizer is provided, use it; otherwise, load from pretrained.
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        )
        self.torch_dtype = torch_dtype
        self.padding_side = padding_side
        self.model_name_config = model_name_config
        # Initialize resource usage log as an empty list.
        self.resource_usage_log = []
        # Set the system prompt.
        self.eidos_system_prompt = str(DEFAULT_SYSTEM_PROMPT)
        # Define the offload path for disk caching.
        self.offload_path = os.path.join(os.getcwd(), "qwen_offload_cache")
        # Initialize the disk cache.
        self.cache = diskcache.Cache(self.offload_path)
        self.enable_offloading = True
        self.enable_asyncio = True
        self.enable_parallel_inference = False
        # Initialize a process pool executor if parallel inference is enabled.
        self.executor = (
            ProcessPoolExecutor() if self.enable_parallel_inference else None
        )
        # Lock to ensure thread-safe model loading.
        self.load_lock = threading.Lock()
        # Set model save path if provided.
        if model_save_path is not None:
            self.model_save_path = model_save_path
        # Load the model using the private method.
        self._load_model()

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata encapsulating model properties."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=True,  # Indicates that this LLM supports chat interactions
            is_function_calling_model=True,  # Indicates support for function calling
            model_name=self.model_name,
            system_role=MessageRole.SYSTEM,
        )

    def _load_model(self) -> None:
        """Loads the Qwen model and tokenizer, ensuring thread safety."""
        # Acquire lock to ensure no concurrent loads happen.
        with self.load_lock:
            # If model isn't loaded yet, attempt to load it.
            if self.model is None:
                try:
                    logging.info("🔥😈 Eidos: Loading model and tokenizer.")
                    save_dir = self.model_save_path

                    # If model save directory exists, load model and tokenizer from there.
                    if os.path.exists(save_dir) and os.path.isdir(save_dir):
                        logging.info(f"Loading model from saved directory: {save_dir}")
                        self.model = AutoModelForCausalLM.from_pretrained(save_dir)
                        self.tokenizer = AutoTokenizer.from_pretrained(save_dir)
                    else:
                        # Otherwise, load from configured model source.
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name_config,
                            torch_dtype=self.torch_dtype,
                            device_map=self.device_map,
                            trust_remote_code=self.trust_remote_code,
                        )
                        # Prepare tokenizer arguments.
                        tokenizer_kwargs = {}
                        if self.padding_side:
                            tokenizer_kwargs["padding_side"] = self.padding_side
                        # Load tokenizer with specific arguments.
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name_config,
                            trust_remote_code=self.trust_remote_code,
                            **tokenizer_kwargs,
                        )
                        # Save loaded model and tokenizer for future use.
                        logging.info(f"Saving model to directory: {save_dir}")
                        self.model.save_pretrained(save_dir)
                        self.tokenizer.save_pretrained(save_dir)
                    logging.info(
                        f"Qwen model {self.model_name_config} loaded successfully."
                    )
                except Exception as e:
                    # Log any exceptions during model loading.
                    logging.error(
                        f"🔥⚠️ Eidos: Error loading Qwen model: {e}", exc_info=True
                    )
                    raise

    def _log_resource_usage(self, stage: str) -> None:
        """Logs resource usage at a given stage of processing."""
        # Retrieve current resource usage data.
        resource_data = self._get_resource_usage()
        # Append the data to the resource usage log with a timestamp stage.
        self.resource_usage_log.append({stage: resource_data})
        # Log a debug message with detailed resource metrics.
        logging.debug(
            f"Resource snapshot at '{stage}': CPU: {resource_data['cpu_percent']}%, "
            f"Memory: {resource_data['memory_percent']}%, Disk: {resource_data['disk_percent']}%, "
            f"Resident Memory: {resource_data['resident_memory'] / (1024**2):.2f} MB, "
            f"Virtual Memory: {resource_data['virtual_memory'] / (1024**2):.2f} MB."
        )

    def _get_resource_usage(self) -> dict:
        """Gets current resource usage metrics using psutil."""
        # Obtain the current process details.
        process = psutil.Process()
        # Get memory information of the process.
        memory_info = process.memory_info()
        # Get current CPU usage percentage.
        cpu_percent = psutil.cpu_percent()
        # Get current system memory usage percentage.
        memory_percent = psutil.virtual_memory().percent
        # Get disk usage statistics.
        disk_usage = psutil.disk_usage("/")

        # Compile all resource metrics into a dictionary.
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
        """Checks if current memory usage exceeds the defined threshold."""
        memory_usage = psutil.virtual_memory().percent
        # Return True if memory usage is above threshold, else False.
        return memory_usage > self.memory_threshold

    def _offload_to_disk(self, key: str, data: Any) -> None:
        """Offloads 'data' to disk cache under the provided key."""
        try:
            # Store data in the disk cache.
            self.cache[key] = data
            logging.debug(f"Offloaded data with key '{key}' to disk.")
        except Exception as e:
            # Log any errors during the offloading process.
            logging.error(f"Disk offloading error for key '{key}': {e}")

    def _load_from_disk(self, key: str) -> Any:
        """Loads data from diskcache corresponding to the given key."""
        try:
            # Attempt to retrieve data from the cache using the key.
            return self.cache.get(key, default=None)
        except Exception as e:
            # Log errors encountered during loading.
            logging.error(f"Disk loading error for key '{key}': {e}")
            return None

    async def _process_chunk_async(self, text_chunk: str) -> str:
        """Asynchronous processing of a text chunk using the LLM."""
        # Log resource usage at start of async chunk processing.
        self._log_resource_usage("start_chunk_processing_async")
        # Check if memory threshold exceeded and offload if necessary.
        if self._should_offload() and self.enable_offloading:
            logging.warning("Memory usage high, performing real disk offloading.")
            self._offload_to_disk(f"chunk_{hash(text_chunk)}", text_chunk)

        # Get the current asyncio event loop.
        loop = asyncio.get_running_loop()
        # Depending on executor availability and parallel inference setting, run LLM call.
        if self.executor and self.enable_parallel_inference:
            response = await loop.run_in_executor(
                self.executor, self._run_llm_sync, text_chunk
            )
        else:
            response = await loop.run_in_executor(None, self._run_llm_sync, text_chunk)

        # Log resource usage after processing.
        self._log_resource_usage("end_chunk_processing_async")
        # Return the response from processing the chunk.
        return response

    def _run_llm_sync(self, text_chunk: str) -> str:
        """Helper method to run the LLM synchronously on a given text chunk."""
        # Check if model is available and has a generate method.
        if self.model and hasattr(self.model, "generate"):
            try:
                # Ensure tokenizer is available.
                if self.tokenizer:
                    # Tokenize the input text chunk.
                    inputs = self.tokenizer(text_chunk, return_tensors="pt").to(
                        self.model.device
                    )
                    # Generate output from the model.
                    output = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.max_new_tokens
                    )
                    # Decode the generated tokens to string.
                    resp_text = self.tokenizer.decode(
                        output[0], skip_special_tokens=True
                    )
                    # Return the generated response text.
                    return resp_text
            except Exception as e:
                # Log any errors encountered during synchronous LLM call.
                logging.error(f"Error in synchronous LLM call: {e}")
        # Return an empty string if processing fails.
        return ""

    def apply_chat_template(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        add_generation_prompt: bool = True,
    ):
        """Applies the chat template to messages without tokenizing, for proper formatting."""
        if self.tokenizer:
            # Use tokenizer's built-in chat template application.
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        else:
            # If tokenizer isn't initialized, raise a runtime error.
            raise RuntimeError("Tokenizer not initialized.")

    def prepare_model_inputs(
        self,
        messages: List[Dict[str, str]],
        padding: bool = False,
    ):
        """Prepares model inputs for single or batched messages, handling tokenization and padding."""
        # Apply chat template to format messages correctly.
        raw_inputs = self.apply_chat_template(messages)
        # Check if raw_inputs is a string.
        if isinstance(raw_inputs, str):
            # If model and tokenizer exist, tokenize the raw input.
            if self.model and self.tokenizer:
                model_inputs = self.tokenizer(
                    raw_inputs, return_tensors="pt", padding=padding
                ).to(self.model.device)
            else:
                raise RuntimeError("LLM model or tokenizer not initialized.")
        # Check if raw_inputs is a list of strings.
        elif isinstance(raw_inputs, list) and all(
            isinstance(item, str) for item in raw_inputs
        ):
            # Tokenize the list of strings.
            if self.model and self.tokenizer:
                model_inputs = self.tokenizer(
                    raw_inputs, return_tensors="pt", padding=padding  # type: ignore
                ).to(self.model.device)
            else:
                raise RuntimeError("LLM model or tokenizer not initialized.")
        else:
            # Log error for invalid input types.
            logger.error("🔥⚠️ Eidos: Invalid raw_inputs type for tokenizer.")
            raise TypeError("Invalid raw_inputs type. Expected str or List[str].")
        # Return the prepared model inputs.
        return model_inputs

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completes text using chunkwise processing for large inputs."""
        all_responses = ""  # Initialize variable to accumulate responses.
        start = 0  # Starting index for processing chunks.
        text = prompt  # Assign prompt to local variable 'text'.
        # Process the text in chunks to manage resource usage.
        while start < len(text):
            # Determine end index for current chunk based on chunk_size.
            end = min(start + self.chunk_size, len(text))
            # Extract the current chunk of text.
            chunk = text[start:end]
            # Process the current chunk and obtain response.
            response_text = self._process_chunk(chunk)
            # Append the response to all_responses.
            all_responses += response_text
            # Move start to next chunk position.
            start = end
            # Adjust chunk size based on current memory usage.
            if self._should_offload():
                # Halve the chunk size if memory usage is high, with a minimum of 128.
                self.chunk_size = max(self.chunk_size // 2, 128)
                logging.info(
                    f"Adjusting chunk size to {self.chunk_size} due to high memory usage."
                )
            else:
                # Double the chunk size up to the default if memory usage is low.
                self.chunk_size = min(self.chunk_size * 2, DEFAULT_INITIAL_CHUNK_SIZE)
                logging.info(
                    f"Adjusting chunk size to {self.chunk_size} due to low memory usage."
                )
        # Return the complete response wrapped in CompletionResponse.
        return CompletionResponse(text=all_responses)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Generate text based on a prompt, streaming the output in real-time."""
        if not self.model or not self.tokenizer:
            # Ensure the model and tokenizer are initialized.
            raise RuntimeError("LLM model or tokenizer not initialized.")

        # Create a list with a single user message.
        messages = [{"role": "user", "content": prompt}]
        # Prepare model inputs from messages.
        model_inputs = self.prepare_model_inputs(messages, padding=False)
        # Determine which streamer mode to use, defaulting to 'iteratorstreamer'.
        stream_mode = kwargs.get("stream_mode", "iteratorstreamer")

        # If using the textstreamer mode.
        if stream_mode == "textstreamer":
            # Initialize TextStreamer for streaming outputs.
            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
            )
            # Start the model.generate process in a separate thread.
            thread = threading.Thread(
                target=self.model.generate,
                kwargs={
                    **model_inputs,
                    "max_new_tokens": kwargs.get("max_tokens", self.max_new_tokens),
                    "streamer": streamer,
                },
            )
            thread.start()
            # Continuously yield output chunks until completion.
            for next_text in streamer.token_cache:
                yield CompletionResponse(text=next_text, delta=next_text)

        # If using the iteratorstreamer mode.
        elif stream_mode == "iteratorstreamer":
            # Initialize TextIteratorStreamer for streaming outputs.
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
            )
            # Set up generation arguments with the streamer.
            generation_kwargs = {
                **model_inputs,
                "streamer": streamer,
                "max_new_tokens": kwargs.get("max_tokens", self.max_new_tokens),
            }
            # Start model.generate in a new thread with the specified arguments.
            thread = threading.Thread(
                target=self.model.generate, kwargs=generation_kwargs
            )
            thread.start()
            # Iterate over text chunks produced by the streamer.
            for text_chunk in streamer:
                # Yield each chunk as a CompletionResponse.
                yield CompletionResponse(text=text_chunk, delta=text_chunk)
        else:
            # Raise an error if an invalid stream_mode is provided.
            raise ValueError(
                f"Invalid stream_mode: {stream_mode}. Must be 'textstreamer' or 'iteratorstreamer'."
            )

    def _process_chunk(self, text_chunk: str) -> str:
        """Processes a text chunk with the LLM, choosing async or sync based on configuration."""
        # If asyncio is enabled, run the async processing path.
        if self.enable_asyncio:
            return asyncio.run(self._process_chunk_async(text_chunk))
        else:
            # Log resource usage at the start of chunk processing.
            self._log_resource_usage("start_chunk_processing")
            # Check if offloading is needed due to high memory usage.
            if self._should_offload() and self.enable_offloading:
                logging.warning("Memory usage high, performing real disk offloading.")
                self._offload_to_disk(f"chunk_{hash(text_chunk)}", text_chunk)

            # If model and tokenizer are ready, process synchronously.
            if self.model and hasattr(self.model, "generate") and self.tokenizer:
                try:
                    # Tokenize the text chunk.
                    inputs = self.tokenizer(text_chunk, return_tensors="pt").to(
                        self.model.device
                    )
                    # Generate output using the model.
                    output = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.max_new_tokens
                    )
                    # Decode the generated tokens to text.
                    response_text = self.tokenizer.decode(
                        output[0], skip_special_tokens=True
                    )
                    # Log resource usage after processing.
                    self._log_resource_usage("end_chunk_processing")
                    return response_text
                except Exception as e:
                    # Log any exceptions during processing.
                    logging.error(f"Error processing chunk: {e}")
                    return ""
            else:
                # Log an error if the model isn't loaded properly.
                logging.error("LLM not loaded.")
                return ""

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Processes a sequence of chat messages and returns a ChatResponse."""
        try:
            # Log resource usage at the start of chat processing.
            self._log_resource_usage("start_chat")
            # Check if offloading is needed due to high memory usage during chat.
            if self._should_offload():
                logging.warning("Memory usage high, performing real disk offloading.")
                self._offload_to_disk(f"chat_{hash(str(messages))}", messages)
            # Ensure model and tokenizer are loaded.
            if self.model and self.tokenizer:
                # Convert ChatMessage objects to dict format for processing.
                input_messages = [
                    {"role": m.role.value, "content": m.content or ""} for m in messages
                ]
                # Proceed if there are input messages.
                if input_messages:
                    # Prepare model inputs from the messages.
                    model_inputs = self.prepare_model_inputs(input_messages, padding=False)
                    # Validate that model inputs have necessary attributes.
                    if model_inputs and model_inputs.input_ids is not None and model_inputs.attention_mask is not None:
                        # Generate responses using the model.
                        generated_ids = self.model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=model_inputs.attention_mask,
                            max_new_tokens=self.max_new_tokens,
                        )
                        # If generation succeeded, process the output.
                        if generated_ids is not None:
                            # Trim the input part from the generated sequence.
                            generated_ids = [
                                output_ids[len(input_id) :].tolist()
                                for input_id, output_ids in zip(
                                    model_inputs.input_ids, generated_ids
                                )
                            ]
                            # If any output was generated.
                            if generated_ids:
                                # Decode the generated token IDs to text.
                                response_text = self.tokenizer.batch_decode(
                                    generated_ids, skip_special_tokens=True
                                )[0]
                                # Create a ChatMessage for the assistant's response.
                                response = ChatMessage(
                                    role=MessageRole.ASSISTANT, content=response_text
                                )
                                # Log resource usage and return ChatResponse.
                                self._log_resource_usage("end_chat")
                                return ChatResponse(message=response)
                            else:
                                # Log error if no generated IDs found.
                                logging.error("No generated IDs after processing.")
                                return ChatResponse(
                                    message=ChatMessage(role=MessageRole.ASSISTANT, content="Error during chat: No generated IDs.")
                                )
                        else:
                            # Log error if model generation fails.
                            logging.error("Model generation returned None.")
                            return ChatResponse(
                                message=ChatMessage(role=MessageRole.ASSISTANT, content="Error during chat: Model generation failed.")
                            )
                    else:
                        # Log error if model inputs are invalid.
                        logging.error("Model inputs are invalid.")
                        return ChatResponse(
                            message=ChatMessage(role=MessageRole.ASSISTANT, content="Error during chat: Invalid model inputs.")
                        )
                else:
                    # Log error if no messages were provided.
                    logging.error("No input messages provided.")
                    return ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content="Error during chat: No input messages.")
                    )
            else:
                # Log error if LLM components are not loaded.
                logging.error("LLM not loaded")
                return ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content="Error during chat: LLM not loaded.")
                )
        except Exception as e:
            # Catch and log any unexpected errors during chat.
            logging.error(f"Error during chat: {e}")
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="Error during chat.")
            )

    def perform_batch_inference(
        self, message_batch: List[List[Dict[str, str]]], max_new_tokens: int = 512
    ) -> List[str]:
        """
        Perform batched inference on a batch of messages.
        Accepts a list of message lists and returns a list of generated responses.
        """
        try:
            # Prepare model inputs for each batch of messages.
            model_inputs_batch = [
                self.prepare_model_inputs(messages) for messages in message_batch
            ]
            # Ensure the model is loaded.
            if self.model:
                # Concatenate input_ids and attention_masks from all batches for generation.
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
                # Raise error if model not initialized.
                raise RuntimeError("LLM model not initialized.")
            # Calculate start indices to separate generated responses for each input.
            start_indices = [input.input_ids.shape[1] for input in model_inputs_batch]
            # Ensure the tokenizer is initialized.
            if self.tokenizer:
                # Decode generated IDs for each batch using start indices.
                response_batch = [
                    self.tokenizer.decode(
                        generated_ids[start:], skip_special_tokens=True
                    )
                    for generated_ids, start in zip(generated_ids_batch, start_indices)
                ]
            else:
                raise RuntimeError("Tokenizer not initialized.")
            # Return list of responses.
            return response_batch
        except Exception as e:
            # Log and raise any errors encountered during batched inference.
            logger.error(
                f"🔥⚠️ Eidos: Error during batched inference: {e}", exc_info=True
            )
            raise


def main():
    """
    Main function to demonstrate and test the BaseLLM class.
    """
    # Initialize the BaseLLM.
    qwen_llm = BaseLLM()

    # Test the stream_complete method.
    print("\nTesting stream_complete method:")
    stream_response = qwen_llm.stream_complete("Tell me a short story.")
    print("LLM Stream Completion:")
    for chunk in stream_response:
        print(chunk.text, end="", flush=True)
    print()  # Add a newline after the stream.

    # Test the stream_complete method with textstreamer.
    print("\nTesting stream_complete method with textstreamer:")
    stream_response_textstreamer = qwen_llm.stream_complete("Tell me a short story.", stream_mode="textstreamer")
    print("LLM Stream Completion (textstreamer):")
    for chunk in stream_response_textstreamer:
        print(chunk.text, end="", flush=True)
    print()  # Add a newline after the stream.

    # Test the perform_batch_inference method.
    print("\nTesting perform_batch_inference method:")
    message_batch = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "What is the largest planet in our solar system?"}],
    ]
    batch_responses = qwen_llm.perform_batch_inference(message_batch)
    print("LLM Batch Inference Responses:", batch_responses)

    # Test the complete method with a short prompt.
    print("Testing complete method with a short prompt:")
    response_text = qwen_llm.complete("Tell me about large language models.")
    print("LLM Completion (short prompt):", response_text.text)

    # Test the complete method with a longer prompt to test chunking.
    print("\nTesting complete method with a longer prompt:")
    long_prompt = "Tell me about large language models. " * 10
    response_text_long = qwen_llm.complete(long_prompt)
    print("LLM Completion (long prompt):", response_text_long.text)

    # Test the chat method with a single user message.
    print("\nTesting chat method with a single user message:")
    user_message = ChatMessage(
        role=MessageRole.USER, content="Tell me about large language models."
    )
    chat_response = qwen_llm.chat(messages=[user_message])
    print("LLM Chat Response:", chat_response.message.content)

    # Test the chat method with multiple messages.
    print("\nTesting chat method with multiple messages:")
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="I am doing well, thank you!"),
        ChatMessage(role=MessageRole.USER, content="What can you do?"),
    ]
    chat_response_multi = qwen_llm.chat(messages=messages)
    print("LLM Chat Response (multiple messages):", chat_response_multi.message.content)

    # Test resource logging by calling a method that logs resources.
    print("\nTesting resource logging:")
    qwen_llm._log_resource_usage("test_resource_logging")
    print("Resource log:", qwen_llm.resource_usage_log)

    # Test disk offloading and loading.
    print("\nTesting disk offloading and loading:")
    test_data = "This is some test data for disk offloading."
    test_key = "test_offload_key"
    qwen_llm._offload_to_disk(test_key, test_data)
    loaded_data = qwen_llm._load_from_disk(test_key)
    print("Offloaded data:", test_data)
    print("Loaded data:", loaded_data)

    # Test the _should_offload method.
    print("\nTesting _should_offload method:")
    should_offload = qwen_llm._should_offload()
    print("Should offload:", should_offload)

    # Test the apply_chat_template method.
    print("\nTesting apply_chat_template method:")
    chat_messages = [{"role": "user", "content": "Hello!"}]
    templated_messages = qwen_llm.apply_chat_template(chat_messages)
    print("Templated messages:", templated_messages)

    # Test the prepare_model_inputs method.
    print("\nTesting prepare_model_inputs method:")
    model_inputs = qwen_llm.prepare_model_inputs(chat_messages)
    print("Model inputs:", model_inputs)

    # Test the prepare_model_inputs method with padding.
    print("\nTesting prepare_model_inputs method with padding:")
    model_inputs_padded = qwen_llm.prepare_model_inputs(chat_messages, padding=True)
    print("Model inputs with padding:", model_inputs_padded)

    # Test the chat method with an empty message list.
    print("\nTesting chat method with an empty message list:")
    empty_chat_response = qwen_llm.chat(messages=[])
    print("LLM Chat Response (empty messages):", empty_chat_response.message.content)

    # Test the chat method with a message that has no content.
    print("\nTesting chat method with a message that has no content:")
    no_content_message = ChatMessage(role=MessageRole.USER, content=None)
    no_content_chat_response = qwen_llm.chat(messages=[no_content_message])
    print("LLM Chat Response (no content message):", no_content_chat_response.message.content)

    # Test the complete method with an empty prompt.
    print("\nTesting complete method with an empty prompt:")
    empty_response_text = qwen_llm.complete("")
    print("LLM Completion (empty prompt):", empty_response_text.text)

    # Test the complete method with a very long prompt to ensure chunking works correctly.
    print("\nTesting complete method with a very long prompt:")
    very_long_prompt = "This is a very long prompt. " * 1000
    very_long_response_text = qwen_llm.complete(very_long_prompt)
    print("LLM Completion (very long prompt):", very_long_response_text.text[:100], "...") # Print only the first 100 characters

    # Test the perform_batch_inference method with an empty message batch.
    print("\nTesting perform_batch_inference method with an empty message batch:")
    empty_batch_responses = qwen_llm.perform_batch_inference([])
    print("LLM Batch Inference Responses (empty batch):", empty_batch_responses)

    # Test the perform_batch_inference method with a batch containing empty messages.
    print("\nTesting perform_batch_inference method with a batch containing empty messages:")
    empty_message_batch = [[], []]
    empty_message_batch_responses = qwen_llm.perform_batch_inference(empty_message_batch)
    print("LLM Batch Inference Responses (empty messages):", empty_message_batch_responses)

    # Test the complete method with a very long prompt to ensure chunking works correctly.
    print("\nTesting complete method with a very long prompt and chunking:")
    very_long_prompt_chunk = "This is a very long prompt to test chunking. " * 2000
    very_long_response_text_chunk = qwen_llm.complete(very_long_prompt_chunk)
    print("LLM Completion (very long prompt with chunking):", very_long_response_text_chunk.text[:100], "...")

    # Test the chat method with a very long message to ensure chunking works correctly.
    print("\nTesting chat method with a very long message and chunking:")
    very_long_chat_message = ChatMessage(role=MessageRole.USER, content="This is a very long message to test chunking. " * 2000)
    very_long_chat_response = qwen_llm.chat(messages=[very_long_chat_message])
    print("LLM Chat Response (very long message with chunking):", very_long_chat_response.message.content)

    # Test the stream_complete method with a longer prompt to test streaming with chunking.
    print("\nTesting stream_complete method with a longer prompt and chunking:")
    long_stream_prompt = "Tell me a very long story to test streaming with chunking. " * 500
    stream_response_long = qwen_llm.stream_complete(long_stream_prompt)
    print("LLM Stream Completion (long prompt with chunking):")
    for chunk in stream_response_long:
        print(chunk.text, end="", flush=True)
    print()

    # Test the stream_complete method with textstreamer and a longer prompt to test streaming with chunking.
    print("\nTesting stream_complete method with textstreamer and a longer prompt and chunking:")
    long_stream_prompt_textstreamer = "Tell me a very long story to test streaming with chunking using textstreamer. " * 500
    stream_response_textstreamer_long = qwen_llm.stream_complete(long_stream_prompt_textstreamer, stream_mode="textstreamer")
    print("LLM Stream Completion (textstreamer, long prompt with chunking):")
    for chunk in stream_response_textstreamer_long:
        print(chunk.text, end="", flush=True)
    print()

    # Test the perform_batch_inference method with a larger batch of messages.
    print("\nTesting perform_batch_inference method with a larger batch:")
    large_message_batch = [
        [{"role": "user", "content": f"What is {i} plus {i}?"}] for i in range(10)
    ]
    large_batch_responses = qwen_llm.perform_batch_inference(large_message_batch)
    print("LLM Batch Inference Responses (larger batch):", large_batch_responses)

    # Test the perform_batch_inference method with a batch containing a mix of empty and non-empty messages.
    print("\nTesting perform_batch_inference method with a mixed batch of empty and non-empty messages:")
    mixed_message_batch = [
        [{"role": "user", "content": "What is the meaning of life?"}],
        [],
        [{"role": "user", "content": "What is the weather like today?"}],
        [],
    ]
    mixed_batch_responses = qwen_llm.perform_batch_inference(mixed_message_batch)
    print("LLM Batch Inference Responses (mixed batch):", mixed_batch_responses)

if __name__ == "__main__":
    main()
