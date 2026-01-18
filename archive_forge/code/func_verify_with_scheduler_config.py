from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
    if scheduler_config.max_num_batched_tokens > 65528:
        raise ValueError('Due to limitations of the custom LoRA CUDA kernel, max_num_batched_tokens must be <= 65528 when LoRA is enabled.')