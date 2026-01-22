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
class CUDAAccelerator(Accelerator):
    """Accelerator for NVIDIA CUDA devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type CUDA.
        """
        if device.type != 'cuda':
            raise ValueError(f'Device should be CUDA, got {device} instead.')
        _check_cuda_matmul_precision(device)
        torch.cuda.set_device(device)

    @override
    def teardown(self) -> None:
        _clear_cuda_memory()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        from lightning_fabric.utilities.device_parser import _parse_gpu_ids
        return _parse_gpu_ids(devices, include_cuda=True)

    @staticmethod
    @override
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device('cuda', i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_cuda_devices()

    @staticmethod
    @override
    def is_available() -> bool:
        return num_cuda_devices() > 0

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register('cuda', cls, description=cls.__name__)