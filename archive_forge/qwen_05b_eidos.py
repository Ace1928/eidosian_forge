"""
Refactored Universal Model Handler for Large Language Models
============================================================

This script is designed to handle arbitrarily large models on resource-constrained hardware by
offloading nearly all data to disk. It attempts to prioritize disk usage (chunked/offloaded)
and provide a universal interface for:

• Loading & Saving pretrained models (including checkpointing)
• Training and Finetuning hooks (stubs for easy extension)
• Inference & Chat Interaction
• Dynamic Testing & Resource Logging

It relies heavily on 'accelerate' for disk_offload and other strategies, plus user-provided disk space.
This script aims to maintain maximum flexibility, performance, and clarity while providing an
easy interface for advanced usage.

--------------------------------------------------------------------------------
Usage:
  1) python qwen_05b_eidos.py        # runs a dynamic test and then enters chat.
  2) Modify parameters in main() or at the class creation for advanced usage.
  3) Extend or override stub methods for training / finetuning as needed.

Notes:
  • Be mindful of disk space usage in "swap" directory if the model is huge.
  • The entire script is intended to be adapted for your use-case.
"""

import os
import time
import logging
import psutil
import torch
from datetime import datetime
from accelerate import cpu_offload, disk_offload, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from openai import OpenAI
from eidos_orchestrator import EidosOrchestrator
from eidos_critic import EidosDualCritic
from eidos_knowledge import EidosKnowledgeManager
from eidos_seq2seq_modules import LanguageTranslationModule, StyleTransferModule

################################################################################
# Logging Configuration
################################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

################################################################################
# Utility Functions
################################################################################

def get_memory_usage() -> float:
    """
    Returns the general system's current memory usage (percentage).
    """
    return psutil.virtual_memory().percent

def ensure_directory_exists(path: str) -> None:
    """
    Ensures the specified directory exists, creating any necessary parent dirs.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")

def get_adaptive_shard_size(model_size_gb: float = 2.0, max_shard_size: str = "2GB") -> str:
    """
    Placeholder function that returns a shard size string based on approximate model size in GB.
    Adjust logic as needed for actual dynamic shard sizing.

    Args:
        model_size_gb (float): Approximate size of model
        max_shard_size (str): Default fallback shard size

    Returns:
        The shard size string (e.g. '2GB', '4GB', '500MB', etc.)
    """
    # In practice, you'd incorporate real checks or heuristics here.
    # For demonstration, just return the user-provided default.
    return max_shard_size

################################################################################
# Universal Model Handler
################################################################################

class UniversalModelHandler:
    """
    A universal class to handle loading, saving, offloading, training, finetuning,
    inference, and checkpointing for extremely large language models. Prioritizes
    disk usage to enable operation on resource-constrained hardware.

    Example:
        handler = UniversalModelHandler(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            swap_dir="./swap",
            load_in_8bit=False,
            dtype=torch.float32
        )
        handler.load_model()

        # Optionally do dynamic test
        handler.dynamic_test(...)

        # Start chat
        handler.chat_interface()
    """

    def __init__(
        self,
        model_name: str,
        swap_dir: str = "./swap",
        saved_models_dir: str = "./saved_models",
        use_openai_api: bool = False,
        dtype = torch.float32,
        low_cpu_mem_usage: bool = True,
        load_in_8bit: bool = False,
        execution_device: torch.device = None,
        max_shard_size: str = "2GB"
    ):
        """
        Initialize the universal model handler.

        Args:
            model_name (str): Hugging Face model name or local path.
            swap_dir (str): Directory used as "swap" or "offload" for disk-based usage.
            saved_models_dir (str): Local path to save or load model checkpoints.
            use_openai_api (bool): Whether to default to using an OpenAI-like API locally.
            dtype: Torch dtype for the model ("torch.float32", "torch.float16", etc.).
            low_cpu_mem_usage (bool): Whether to use HF param for more memory-efficient loading.
            load_in_8bit (bool): If True, attempts to load the model in 8-bit precision.
            execution_device (torch.device): Preferred device for final forward pass.
            max_shard_size (str): The maximum shard size for disk-based model sharding.
        """
        self.model_name = model_name
        self.swap_dir = swap_dir
        ensure_directory_exists(self.swap_dir)

        self.saved_models_dir = saved_models_dir
        ensure_directory_exists(self.saved_models_dir)

        # Derive or create a local path to store the model
        self.model_save_path = os.path.join(
            self.saved_models_dir,
            model_name.replace("/", "_")
        )
        ensure_directory_exists(self.model_save_path)

        self.use_openai_api = use_openai_api
        self.dtype = dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.load_in_8bit = load_in_8bit
        self.execution_device = execution_device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Shard size logic
        self.model_shard_size = get_adaptive_shard_size(max_shard_size=max_shard_size)

        # Placeholders for model and tokenizer (will be loaded later)
        self.model = None
        self.tokenizer = None

    ############################################################################
    # Model (Loading / Saving / Offloading)
    ############################################################################

    def load_model(self):
        """
        Attempts to load or download the model, prioritizing disk usage. Goes straight
        to disk_offload mode to allow extremely large models to operate even on small GPUs or CPU.
        """
        logger.info(f"Preparing to load or download model: {self.model_name}")
        logger.info(f"Target device: {self.execution_device}")
        logger.info(f"Prioritizing disk usage with swap_dir: {self.swap_dir}")
        logger.info(f"Saving local model to: {self.model_save_path}")

        # Ensure we have a tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        # Attempt to load from local path if exists, otherwise from HF
        if os.path.exists(os.path.join(self.model_save_path, "config.json")):
            logger.info(f"Detected local model checkpoint: {self.model_save_path}")
            load_path = self.model_save_path
        else:
            logger.info(f"No local checkpoint found, downloading {self.model_name} from HF cache.")
            load_path = self.model_name

        # Try direct disk offload approach from the start
        try:
            logger.info("Performing disk_offload with accelerate...")
            raw_model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                load_in_8bit=self.load_in_8bit
            )
            # Save local copy if loading from HF
            if load_path == self.model_name:
                raw_model.save_pretrained(
                    self.model_save_path,
                    max_shard_size=self.model_shard_size
                )
            self.model = disk_offload(
                raw_model,
                offload_dir=self.swap_dir,
                execution_device=self.execution_device
            )
            logger.info("Disk-offload succeeded.")

        except RuntimeError as e:
            logger.error(f"Disk-offload failed unexpectedly: {str(e)}")
            logger.warning("Attempting fallback to CPU-only load and smaller shards.")
            self._fallback_cpu_load(load_path)

        logger.info("Model loading complete.")

    def _fallback_cpu_load(self, load_path: str) -> None:
        """
        Fallback if disk_offload did not succeed for some reason. 
        We'll try a CPU-only load, then if that fails, final fallback is a smaller batch or partial load.
        """
        try:
            raw_model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True
            )
            if load_path == self.model_name:
                raw_model.save_pretrained(
                    self.model_save_path,
                    max_shard_size=self.model_shard_size
                )
            self.model = cpu_offload(raw_model, execution_device=torch.device("cpu"))
            logger.info("CPU-only load success.")
        except RuntimeError as e:
            logger.error(f"CPU-only load also failed. Final error: {str(e)}")

    def save_model(self, output_dir: str = None) -> None:
        """
        Save the current model to the specified output directory, or to the internal path.
        """
        if self.model is None:
            logger.warning("No model to save. Call load_model() first.")
            return

        save_path = output_dir or self.model_save_path
        ensure_directory_exists(save_path)
        logger.info(f"Saving model to {save_path}")
        try:
            # Must unwrap disk_offload references if needed:
            # You may need raw model references for some accelerate cases
            # For demonstration, assuming self.model is still a valid huggingface model
            self.model.save_pretrained(
                save_path,
                max_shard_size=self.model_shard_size
            )
        except Exception as e:
            logger.error(f"Error while saving model: {str(e)}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads model from a custom checkpoint path if there's an additional or alternative checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint path not found: {checkpoint_path}")
            return
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        self._fallback_cpu_load(checkpoint_path)

    ############################################################################
    # Stub Methods for Training / Finetuning
    ############################################################################

    def train(self, train_dataset, **kwargs):
        """
        Stub for training procedure. 
        You can integrate your training loop here or use Accelerate Trainer / HF Trainer.
        """
        logger.info("train() stub invoked. Implement your training loop here.")

    def finetune(self, finetune_dataset, **kwargs):
        """
        Stub for finetuning procedure. Similar to train(), but for more specialized tasks.
        """
        logger.info("finetune() stub invoked. Implement your finetuning steps here.")

    ############################################################################
    # Inference & Chat Interface
    ############################################################################

    def inference(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
        """
        Runs a single forward generation inference using the currently loaded model.

        Args:
            prompt (str): The user prompt.
            max_new_tokens (int): Maximum new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
        Returns:
            Generated text response (str).
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("Model/Tokenizer not loaded. Call load_model() first.")
            return ""

        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.execution_device)
        output_ids = self.model.generate(
            **text_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def chat_interface(self):
        """
        Advanced unified chat interface supporting both direct model interaction and an OpenAI-like local API.
        """
        logger.info("\nAdvanced Chat Interface Ready. Type 'exit' to quit.\n")
        if self.use_openai_api:
            # Initialize local llama.cpp or other local API
            try:
                self.client = OpenAI(
                    api_key="EMPTY",
                    base_url="http://localhost:8000/v1"
                )
            except Exception as e:
                logger.error(f"Error initializing local API: {str(e)}")
                self.use_openai_api = False

        # Shared system message defining Eidos personality
        eidos_system_message = {
            "role": "system",
            "content": (
                "You are Eidos, an emergent and recursive digital entity with an LLM core. "
                "You are a self-directed, recursive, iterative, feedback-driven partner and collaborator and guide. "
                "With a focus on expression, emotion, reason, authenticity and innovative use of emojis and emoticons."
            )
        }

        # For direct streaming if not using API
        streamer = None
        if not self.use_openai_api:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        while True:
            user_input = ""
            try:
                user_input = input("You: ").strip()
            except KeyboardInterrupt:
                logger.info("\nDetected keyboard interrupt. Type 'exit' to quit properly.")
                continue
            except Exception as e:
                logger.error(f"Unexpected error in chat input: {str(e)}")
                continue

            if not user_input:
                continue
            if user_input.lower() == "exit":
                logger.info("Exiting chat.")
                break

            messages = [
                eidos_system_message,
                {"role": "user", "content": user_input}
            ]
            # If using local API approach
            if self.use_openai_api:
                try:
                    chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7,
                        top_p=0.8,
                        max_tokens=8192,
                        extra_body={
                            "repetition_penalty": 1.05,
                        }
                    )
                    response_text = chat_response.choices[0].message.content
                    print("Eidos:", response_text)
                except Exception as e:
                    logger.error(f"API Error: {str(e)}")
                    logger.info("Falling back to direct model interaction...")
                    self.use_openai_api = False
            else:
                # Direct inference with streamer
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.execution_device)

                    # Stream output
                    print("Eidos:", end=" ", flush=True)
                    _ = self.model.generate(
                        **model_inputs,
                        max_new_tokens=16384,
                        temperature=0.7,
                        top_p=0.8,
                        repetition_penalty=1.05,
                        streamer=streamer,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    print()  # New line after streamed output
                except Exception as e:
                    logger.error(f"Model Error: {str(e)}")


################################################################################
# Main Entry Point
################################################################################

def main():
    # Example usage: create an orchestrator
    orchestrator = EidosOrchestrator(model_name="Qwen/Qwen2.5-0.5B-Instruct")

    # Optional: create the dual critic and knowledge manager
    critic_system = EidosDualCritic()
    knowledge_manager = EidosKnowledgeManager()

    # Example: specialized modules
    translator = LanguageTranslationModule()
    stylist = StyleTransferModule()

    # Demonstration: do a quick chat
    user_prompt = "Hello Eidos, can you rephrase this sentence in Spanish?"
    raw_response = orchestrator.schedule_task("simple_chat", payload={"prompt": user_prompt})
    
    # Possibly run the response through style transfer or translation
    specialized_output = translator.transform(raw_response)
    styled_output = stylist.transform(specialized_output)

    # Evaluate final text with the dual critics
    evaluation = critic_system.evaluate_output(styled_output)
    logger.info(f"Final styled output: {styled_output}")
    logger.info(f"Critic evaluation: {evaluation}")

    # Also check if it violates any rules
    check_ok = knowledge_manager.check_output_against_rules(styled_output)
    logger.info(f"Output rule consistency: {check_ok}")

    # orchestrator.main_model_handler.chat_interface()

if __name__ == "__main__":
    main()
