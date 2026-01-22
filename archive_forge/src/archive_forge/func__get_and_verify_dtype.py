from typing import Optional, Union, ClassVar
from dataclasses import dataclass
import os
from packaging.version import Version
import torch
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip, is_neuron, get_nvcc_cuda_version
def _get_and_verify_dtype(config: PretrainedConfig, dtype: Union[str, torch.dtype]) -> torch.dtype:
    config_dtype = getattr(config, 'torch_dtype', None)
    if config_dtype is None:
        config_dtype = torch.float32
    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == 'auto':
            if config_dtype == torch.float32:
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f'Unknown dtype: {dtype}')
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f'Unknown dtype: {dtype}')
    if is_hip() and torch_dtype == torch.float32:
        rocm_supported_dtypes = [k for k, v in _STR_DTYPE_TO_TORCH_DTYPE.items() if k not in _ROCM_NOT_SUPPORTED_DTYPE]
        raise ValueError(f"dtype '{dtype}' is not supported in ROCm. Supported dtypes are {rocm_supported_dtypes}")
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            pass
        elif config_dtype == torch.float32:
            pass
        else:
            logger.warning(f'Casting {config_dtype} to {torch_dtype}.')
    return torch_dtype