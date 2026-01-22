import platform
from functools import lru_cache
from typing import List, Optional, Union
import torch
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
class MPSAccelerator(Accelerator):
    """Accelerator for Metal Apple Silicon GPU devices.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not MPS.
        """
        if device.type != 'mps':
            raise ValueError(f'Device should be MPS, got {device} instead.')

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        from lightning_fabric.utilities.device_parser import _parse_gpu_ids
        return _parse_gpu_ids(devices, include_mps=True)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, str, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        parsed_devices = MPSAccelerator.parse_devices(devices)
        assert parsed_devices is not None
        return [torch.device('mps', i) for i in range(len(parsed_devices))]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    @override
    @lru_cache(1)
    def is_available() -> bool:
        """MPS is only available on a machine with the ARM-based Apple Silicon processors."""
        return torch.backends.mps.is_available() and platform.processor() in ('arm', 'arm64')

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register('mps', cls, description=cls.__name__)