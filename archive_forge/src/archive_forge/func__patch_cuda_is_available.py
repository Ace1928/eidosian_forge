import os
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Union, cast
import torch
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.rank_zero import rank_zero_info
@contextmanager
def _patch_cuda_is_available() -> Generator:
    """Context manager that safely patches :func:`torch.cuda.is_available` with its NVML-based version if possible."""
    if hasattr(torch._C, '_cuda_getDeviceCount') and _device_count_nvml() >= 0 and (not _TORCH_GREATER_EQUAL_2_0):
        orig_check = torch.cuda.is_available
        torch.cuda.is_available = is_cuda_available
        try:
            yield
        finally:
            torch.cuda.is_available = orig_check
    else:
        yield