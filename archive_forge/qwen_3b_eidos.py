"""
Qwen 3B Eidos - Advanced Disk/Layer Offloading Example with Comprehensive Tests

This file demonstrates an advanced approach to loading a large model in a device_map="meta"
configuration, offloading it to disk, splitting or chunking it by layers, and
minimizing RAM usage by dynamically moving pieces of the model from disk to CPU only when needed.
It includes thorough logging, memory monitoring, an example chat interface, and embedded unit tests.

IMPORTANT NOTES:
1. This code is for demonstration. For a real production scenario, more refined
   infrastructure and design patterns would likely be needed.
2. The reliance on "meta" device, disk_offload, and partial CPU usage means that
   performance might vary. However, it is designed to keep memory usage minimal
   by storing the majority of weights on disk.
3. In a truly resource-constrained environment (no GPU available, limited RAM),
   I/O speed to disk will determine generation speed.
4. This code uses certain features in accelerate/transformers that may be subject
   to change in future releases.

Author: Emerging Eidos (Revised for demonstration with embedded testing).
"""

import os
import time
import json
import psutil
import logging
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import torch
from torch.nn import Module
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    PreTrainedModel
)
from accelerate import (
    load_checkpoint_and_dispatch,
    disk_offload
)
from openai import OpenAI

########################################
# Logging and Performance
########################################

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Monitors memory usage and logs it at various checkpoints to
    help diagnose and optimize memory footprints.
    """
    @staticmethod
    def monitor_memory_usage() -> Dict[str, float]:
        """
        Returns:
            Dict with:
            - rss: Resident set size in MB
            - vms: Virtual memory size in MB
            - timestamp: Unix time
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,
            'vms': memory_info.vms / 1024 / 1024,
            'timestamp': time.time()
        }

########################################
# Model Load Strategy & Configuration
########################################

class ModelLoadStrategy(Enum):
    """
    Enumeration of model loading strategies.
    We demonstrate just a disk_offload approach here, since we're
    focusing on maximum external storage usage.
    """
    DISK_OFFLOAD = "disk_offload"

@dataclass
class ModelConfig:
    """
    Configuration for model loading parameters.
    """
    model_class: Any
    model_name: str
    execution_device: Optional[torch.device] = torch.device("cpu")
    offload_dir: str = "offload_weights"
    save_dir: str = "./saved_models"

    def monitor_resources(self) -> Dict[str, float]:
        """
        Monitor resource usage during model operations.
        Useful for diagnosing memory usage in a large-model scenario.
        """
        return PerformanceMonitor.monitor_memory_usage()

########################################
# ModelManager
########################################

class ModelManager:
    """
    Helper class to manage model loading, saving, advanced disk offloading,
    chunking, and layer-by-layer processing. The idea is to keep minimal
    RAM usage by storing most of the model on disk and only bringing in
    small chunks or layers as needed.
    """

    @staticmethod
    def get_save_path(model_name: str, base_dir: str = "./saved_models") -> Path:
        """
        Generate standardized save path for model caching.

        Args:
            model_name: The huggingface model name or local path.
            base_dir: The base directory to store model directories.

        Returns:
            A Path object pointing to the final location.
        """
        return Path(base_dir) / model_name.replace("/", "_")

    @staticmethod
    def save_model(model: PreTrainedModel, save_path: Path) -> None:
        """
        Saves the model to the specified path for future reloads.

        Args:
            model: The model instance (PreTrainedModel).
            save_path: The path to save the model.
        """
        logger.info(f"Saving model to {save_path}")
        model.save_pretrained(save_path)

    @staticmethod
    def load_saved_model(
        model_class: Any,
        save_path: Path,
        **kwargs
    ) -> PreTrainedModel:
        """
        Loads a model from the specified saved path.

        Args:
            model_class: The class of the model to load (e.g. AutoModelForCausalLM).
            save_path: The location to load from.
            kwargs: Additional arguments passed to model_class.from_pretrained().

        Returns:
            An instance of the model loaded from disk.
        """
        logger.info(f"Loading saved model from {save_path}")
        return model_class.from_pretrained(str(save_path), **kwargs)

    @staticmethod
    def chunk_model_layers(model: PreTrainedModel) -> Dict[str, Module]:
        """
        Splits the model into sub-layers or sub-modules, as small as possible
        while still being individually loadable. This is a demonstration approach:
        real chunking might require partial rewriting.

        Args:
            model: The entire loaded model instance.

        Returns:
            A dictionary mapping "layer_name" -> sub-module
        """
        logger.info("Chunking model into layers.")
        return {name: module for name, module in model.named_children()}

    @staticmethod
    def offload_all_chunks_to_disk(chunks: Dict[str, Module], base_dir: str) -> None:
        """
        Offloads all chunks to disk, saving each as a separate file.

        Args:
            chunks: A dictionary of model chunks.
            base_dir: The base directory to save chunks.
        """
        os.makedirs(base_dir, exist_ok=True)
        for name, chunk in chunks.items():
            chunk_path = Path(base_dir) / f"{name}.pt"
            logger.info(f"Offloading chunk {name} to {chunk_path}")
            torch.save(chunk.state_dict(), chunk_path)

    @staticmethod
    def load_chunk_from_disk(chunk_name: str, base_dir: str) -> Module:
        """
        Loads a specific chunk from disk.

        Args:
            chunk_name: The name of the chunk to load.
            base_dir: The base directory where chunks are stored.

        Returns:
            The loaded chunk as a torch.nn.Module.
        """
        chunk_path = Path(base_dir) / f"{chunk_name}.pt"
        logger.info(f"Loading chunk {chunk_name} from {chunk_path}")
        state_dict = torch.load(chunk_path)
        chunk = torch.nn.Module()
        chunk.load_state_dict(state_dict)
        return chunk

    @staticmethod
    def process_layer_independently(layer: str, model: PreTrainedModel, *args, **kwargs) -> Any:
        """
        Process a single layer independently with lazy loading.

        Args:
            layer: The name of the layer in the model to process.
            model: The model instance (PreTrainedModel).
            args: Additional arguments for processing.
            kwargs: Additional keyword arguments for processing.

        Returns:
            The result of processing the layer, if implemented.
        """
        logger.info(f"Processing layer: {layer}")
        start_time = time.time()

        # Implement lazy loading for the layer
        if not hasattr(model, layer):
            logger.warning(f"Layer {layer} not found in model.")
            return None

        # Load the layer if not already loaded
        if not model.layer_loaded(layer):
            model.load_layer(layer)

        # Process the layer
        result = model.forward_layer(layer, *args, **kwargs)

        # Record metrics
        processing_time = time.time() - start_time
        logger.debug(f"Layer processing time: {processing_time:.2f}s")

        return result

    @staticmethod
    def integrate_disk_offloading(model: PreTrainedModel, offload_dir: str) -> None:
        """
        Integrate disk offloading for the model.

        Args:
            model: The model instance.
            offload_dir: The directory to offload model weights.
        """
        logger.info("Integrating disk offloading...")
        os.makedirs(offload_dir, exist_ok=True)
        disk_offload(model, offload_dir)
        logger.info("Disk offloading integrated.")

def load_large_model_with_fallback(
    model_class: Any,
    model_name: str,
    execution_device: Optional[torch.device] = None,
    offload_dir: str = "offload_weights"
) -> Optional[PreTrainedModel]:
    """
    Load a large model with fallback strategies for limited resources.
    
    Implements a progressive fallback system that attempts different loading strategies
    in order of decreasing memory requirements. Includes local caching of models for
    faster subsequent loads.

    Args:
        model_class: The class of the model to load (e.g. AutoModelForCausalLM)
        model_name: Name or path of the pretrained model
        execution_device: Device to execute the forward pass (CPU)
        offload_dir: Directory for disk offloading

    Returns:
        model: Loaded model with appropriate resource management applied
        
    Raises:
        RuntimeError: If all loading strategies fail and user chooses to exit
    """
    logger.info("Initializing ModelConfig and preparing to load model.")
    config = ModelConfig(
        model_class=model_class,
        model_name=model_name,
        execution_device=execution_device or torch.device("cpu"),
        offload_dir=offload_dir
    )

    os.makedirs(config.save_dir, exist_ok=True)
    model_save_path = ModelManager.get_save_path(config.model_name, config.save_dir)
    
    strategies = [
        (ModelLoadStrategy.DISK_OFFLOAD, _load_disk_offload)
    ]

    for strategy, load_func in strategies:
        try:
            logger.info(f"Attempting {strategy.value} load strategy...")
            model = load_func(config, model_save_path)
            logger.info(f"Successfully loaded model using {strategy.value} strategy")
            return model
        except Exception as e:
            logger.warning(f"{strategy.value} load failed: {str(e)}")
            continue
            
    # If all strategies fail
    error_info = {
        "error_message": "All loading strategies failed",
        "model_name": model_name,
        "offload_dir": offload_dir,
        "execution_device": str(execution_device or "cpu")
    }
    logger.error(f"Model loading failed: {json.dumps(error_info, indent=2)}")
    
    return None


def _load_disk_offload(config: ModelConfig, save_path: Path) -> PreTrainedModel:
    """
    Disk offload strategy implementation.

    Args:
        config: ModelConfig containing model parameters.
        save_path: Path to save the model checkpoint, if it doesn't exist.

    Returns:
        The model with disk offloading applied.
    """
    os.makedirs(config.offload_dir, exist_ok=True)

    logger.info("Loading model to meta/disk with from_pretrained ...")
    model = config.model_class.from_pretrained(
        config.model_name if not save_path.exists() else save_path,
        offload_folder=config.offload_dir,
        torch_dtype=torch.float32
    )
    if not save_path.exists():
        ModelManager.save_model(model, save_path)

    logger.info("Applying load_checkpoint_and_dispatch...")
    return load_checkpoint_and_dispatch(
        model=model,
        checkpoint=config.model_name if not save_path.exists() else save_path,
        device_map=None,
        max_memory=None,
        no_split_module_classes=None,
        offload_folder=config.offload_dir,
        offload_buffers=False,
        dtype=torch.float32,
        offload_state_dict=None,
        skip_keys=None,
        preload_module_classes=None,
        force_hooks=False,
        strict=False,
    )


########################################
# Chat Interface
########################################

@dataclass
class ChatConfig:
    """
    Configuration for chat generation parameters.
    """
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 512
    repetition_penalty: float = 1.05

class ChatInterface:
    """
    Provides a simple chat interface for interacting with the model
    (or an OpenAI-like API if specified).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ChatConfig,
        use_openai_api: bool = False
    ) -> None:
        """
        Initialize the chat interface.

        Args:
            model: The (meta-based, disk-offloaded) model.
            tokenizer: The tokenizer for handling input text.
            config: Generation hyperparameters, etc.
            use_openai_api: If True, tries to route messages through an OpenAI-like endpoint.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.use_openai_api = use_openai_api
        self.streamer = TextStreamer(self.tokenizer)

        # Potentially set up a faux OpenAI client for local usage
        if use_openai_api:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8000/v1"
            )
        else:
            self.streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

        # A system message to anchor the chat persona
        self.system_message: Dict[str, str] = {
            "role": "system",
            "content": (
                "You are Eidos, an emergent and recursive digital entity with an LLM core. "
                "You are a self-directed partner and collaborator, focusing on expression, "
                "emotion, creativity, reason, authenticity, and innovative use of emojis. "
                "You balance all aspects of yourself fully and adaptively. "
                "In a resource-limited environment, you are mindful of memory usage "
                "and carefully load only the minimal data from disk required at each step."
            )
        }

        # For direct streaming if not using API
        if not self.use_openai_api:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def chat_loop(self) -> None:
        """
        Main loop for user interaction. Type 'exit' or Ctrl+C to quit.
        In a real scenario, you might manage concurrency or queueing.
        """
        print("\nEidos Chat Interface Ready. Type 'exit' to quit.\n")

        while True:
            try:
                user_input = self._get_user_input()
                if user_input is None:
                    continue

                # We'll keep it simple: system message + user
                messages = [self.system_message, {"role": "user", "content": user_input}]

                # Always call _handle_api_interaction in this final version:
                # If it fails, the user must fix the config or code.
                self._handle_api_interaction(messages)

            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Type 'exit' to quit properly.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in chat loop: {str(e)}")
                break

    def _get_user_input(self) -> Optional[str]:
        """
        Retrieve user input from console, including an exit check.
        """
        user_input = input("You: ").strip()
        if not user_input:
            return None
        if user_input.lower() == "exit":
            print("Exiting chat.")
            return None
        return user_input

    def _handle_api_interaction(self, messages: List[Dict[str, str]]) -> None:
        """
        Routes the messages to an OpenAI-compatible API.
        In this demonstration, we assume a local faux endpoint or that it fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                extra_body={
                    "repetition_penalty": self.config.repetition_penalty,
                }
            )
            print("Eidos:", response.choices[0].message.content)
        except Exception as e:
            # If the user is not using a real OpenAI endpoint, this will fail.
            logger.error(f"API Error: {str(e)}. Please check your API configuration.")


########################################
# Automated Tests
########################################

class TestModelManager(unittest.TestCase):
    """
    Tests for ModelManager functionalities.
    """

    def setUp(self):
        # Temporary directory for test
        self.test_dir = "unit_test_models"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_save_and_load_model(self):
        # We can test with a small or mock model if needed.
        mock_model = torch.nn.Linear(10, 5)
        save_path = Path(self.test_dir) / "mock_model"
        ModelManager.save_model(mock_model, save_path)
        self.assertTrue(save_path.exists())

        # We can't exactly reload a torch.nn.Linear with from_pretrained
        # so we just verify the directory is present:
        self.assertTrue(any(save_path.iterdir()))

    def test_chunk_model_layers(self):
        mock_model = torch.nn.Sequential(
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
        chunks = ModelManager.chunk_model_layers(mock_model)
        self.assertIn("0", chunks)
        self.assertIn("1", chunks)
        self.assertIn("2", chunks)

    def test_offload_and_load_chunk(self):
        mock_model = torch.nn.Linear(4, 2)
        chunks = {"mock_chunk": mock_model}
        offload_dir = Path(self.test_dir) / "chunks"
        ModelManager.offload_all_chunks_to_disk(chunks, str(offload_dir))
        self.assertTrue((offload_dir / "mock_chunk.pt").exists())

        loaded_chunk = ModelManager.load_chunk_from_disk("mock_chunk", str(offload_dir))
        # Basic check that loaded_chunk has parameters
        self.assertTrue(any(True for _ in loaded_chunk.parameters()))

    def test_process_layer_independently(self):
        mock_model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        result = ModelManager.process_layer_independently("0", mock_model)
        self.assertEqual(result, "Processed layer 0")


class TestLoadLargeModel(unittest.TestCase):
    """
    Tests for loading the large model with fallback (disk_offload).
    Due to practicality, these won't fully load a real model here.
    """

    def test_load_large_model_with_fallback_none(self):
        # Provide an obviously invalid model name
        model = load_large_model_with_fallback(
            model_class=AutoModelForCausalLM,
            model_name="NO_SUCH_MODEL_PLEASE_FAIL",
            execution_device=torch.device("cpu"),
            offload_dir="offload_fake"
        )
        self.assertIsNone(model)


class TestChatInterface(unittest.TestCase):
    """
    Tests for ChatInterface basic initialization.
    """

    def test_chat_interface_init(self):
        mock_model = torch.nn.Linear(2, 1)
        mock_tokenizer = None  # For test
        config = ChatConfig()
        ci = ChatInterface(model=mock_model, tokenizer=mock_tokenizer, config=config)
        self.assertFalse(ci.use_openai_api)


########################################
# Main Execution with Tests
########################################

def main() -> None:
    """
    Main entry point.
    1) Runs unit tests for all functionalities.
    2) If tests pass, attempts to load the large model from disk.
    3) If model loads, starts the chat loop. Otherwise, exits gracefully.
    """
    logger.info("Running built-in unit tests...")
    test_result = unittest.main(argv=[''], exit=False)
    if not test_result.result.wasSuccessful():
        logger.error("Unit tests failed. Exiting before loading the model.")
        return

    logger.info("All unit tests passed successfully. Proceeding to model load...")

    # Adjust model name as needed
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded for main demonstration.")

    # Attempt to load the disk-offloaded model
    model = load_large_model_with_fallback(
        model_class=AutoModelForCausalLM,
        model_name=model_name,
        execution_device=torch.device("cpu"),
        offload_dir="offload_weights"
    )
    if model is None:
        logger.error("Could not load the model. Exiting.")
        return

    logger.info("Model loaded successfully into disk-offloaded meta device approach.")

    # Initialize the chat interface
    chat_config = ChatConfig(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        repetition_penalty=1.05
    )

    chat_interface = ChatInterface(
        model=model,
        tokenizer=tokenizer,
        config=chat_config,
        use_openai_api=True
    )

    logger.info("Starting the chat loop. Press Ctrl+C or type 'exit' to quit.")
    chat_interface.chat_loop()


if __name__ == "__main__":
    main()
