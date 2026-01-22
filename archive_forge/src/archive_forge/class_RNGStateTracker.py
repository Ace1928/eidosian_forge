import contextlib
import warnings
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
class RNGStateTracker:
    """
    RNGStateTracker stores Random Number Generator (RNG) state (a ByteTensor object)
    in a dict, mapping from a corresponding tag to each state tensor. It also provides
    a set of convenient utility methods to help access/modify the state tensors. The most
    important interface is _distribute_region which will be used when DTensor executes
    a random op (an operator that calls RNG).
    """

    def __init__(self, device_type: str='cuda'):
        self._device_type = device_type
        self._device_handle = _get_device_handle(device_type)
        if not (self._device_handle and self._device_handle.is_available()):
            raise RuntimeError(f'{self.__class__.__name__} instantiation requires the presence of CUDA/CUDA-like device')
        self._states: Dict[str, Tensor] = {}
        self._devices = [self._device_handle.current_device()]
        self._use_distribute_region = True

    @property
    def rng_states(self) -> Dict[str, Tensor]:
        return self._states

    @property
    def distribute_region_enabled(self) -> bool:
        return self._use_distribute_region

    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value) -> None:
        self._use_distribute_region = value

    def rng_state_is_sync(self, name) -> bool:
        return name in self.rng_states

    def get_seed(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(f'{self.__class__.__name__} does not have random state for {name}')
        seed_tensor = self.rng_states[name][0:8].view(dtype=torch.int64)
        return int(seed_tensor.item())

    def set_seed(self, name: str, seed: int) -> None:
        seed_tensor = torch.tensor([seed]).view(torch.uint8)
        offset_tensor = torch.tensor([0]).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

    def _distribute_region(self, spec: DTensorSpec):
        pass