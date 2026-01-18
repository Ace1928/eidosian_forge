import gc
import itertools
import tempfile
import unittest
import torch
from accelerate.utils.memory import release_memory
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, is_peft_available
from trl.models.utils import setup_chat_format
from ..testing_utils import require_bitsandbytes, require_peft, require_torch_gpu, require_torch_multi_gpu
from .testing_constants import DEVICE_MAP_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS, MODELS_TO_TEST, PACKING_OPTIONS

        Simply tests if using setup_chat_format with a transformers model + peft + bnb config to `SFTTrainer` loads and runs the trainer
        as expected.
        