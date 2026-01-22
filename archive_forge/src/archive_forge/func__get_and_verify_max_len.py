from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def _get_and_verify_max_len(hf_config: PretrainedConfig, max_model_len: Optional[int]) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float('inf')
    possible_keys = ['max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_length', 'max_sequence_length', 'max_seq_length', 'seq_len']
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)
    if derived_max_model_len == float('inf'):
        if max_model_len is not None:
            return max_model_len
        default_max_len = 2048
        logger.warning(f"The model's config.json does not contain any of the following keys to determine the original maximum length of the model: {possible_keys}. Assuming the model's maximum length is {default_max_len}.")
        derived_max_model_len = default_max_len
    rope_scaling = getattr(hf_config, 'rope_scaling', None)
    if rope_scaling is not None:
        assert 'factor' in rope_scaling
        scaling_factor = rope_scaling['factor']
        if rope_scaling['type'] == 'yarn':
            derived_max_model_len = rope_scaling['original_max_position_embeddings']
        derived_max_model_len *= scaling_factor
    if max_model_len is None:
        max_model_len = derived_max_model_len
    elif max_model_len > derived_max_model_len:
        raise ValueError(f"User-specified max_model_len ({max_model_len}) is greater than the derived max_model_len ({max_len_key}={derived_max_model_len} in model's config.json). This may lead to incorrect model outputs or CUDA errors. Make sure the value is correct and within the model context size.")
    return int(max_model_len)