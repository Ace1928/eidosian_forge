from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def _verify_load_format(self) -> None:
    load_format = self.load_format.lower()
    supported_load_format = ['auto', 'pt', 'safetensors', 'npcache', 'dummy']
    rocm_not_supported_load_format = []
    if load_format not in supported_load_format:
        raise ValueError(f"Unknown load format: {self.load_format}. Must be one of 'auto', 'pt', 'safetensors', 'npcache', or 'dummy'.")
    if is_hip() and load_format in rocm_not_supported_load_format:
        rocm_supported_load_format = [f for f in supported_load_format if f not in rocm_not_supported_load_format]
        raise ValueError(f"load format '{load_format}' is not supported in ROCm. Supported load format are {rocm_supported_load_format}")
    architectures = getattr(self.hf_config, 'architectures', [])
    if 'MixtralForCausalLM' in architectures and load_format == 'pt':
        raise ValueError("Currently, the 'pt' format is not supported for Mixtral. Please use the 'safetensors' format instead. ")
    self.load_format = load_format