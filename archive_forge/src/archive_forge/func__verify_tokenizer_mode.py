from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def _verify_tokenizer_mode(self) -> None:
    tokenizer_mode = self.tokenizer_mode.lower()
    if tokenizer_mode not in ['auto', 'slow']:
        raise ValueError(f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be either 'auto' or 'slow'.")
    self.tokenizer_mode = tokenizer_mode