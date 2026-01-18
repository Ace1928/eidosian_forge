import gc
import itertools
import tempfile
import unittest
import torch
from accelerate.utils.memory import release_memory
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer, is_peft_available
from ..testing_utils import require_bitsandbytes, require_peft, require_torch_gpu
from .testing_constants import DPO_LOSS_TYPES, DPO_PRECOMPUTE_LOGITS, GRADIENT_CHECKPOINTING_KWARGS, MODELS_TO_TEST

        A test that tests the simple usage of `DPOTrainer` using QLoRA + different scenarios of gradient checkpointing.
        