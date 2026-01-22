from __future__ import annotations
from typing import Any
import torch
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.strategy import Strategy, TBroadcast
from lightning_fabric.utilities.types import _DEVICE
class SingleDeviceStrategy(Strategy):
    """Strategy that handles communication on a single device."""

    def __init__(self, device: _DEVICE='cpu', accelerator: Accelerator | None=None, checkpoint_io: CheckpointIO | None=None, precision: Precision | None=None):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision=precision)
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self._root_device = device
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    @property
    @override
    def root_device(self) -> torch.device:
        return self._root_device

    @property
    @override
    def is_global_zero(self) -> bool:
        return True

    @override
    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    @override
    def all_reduce(self, tensor: Any | torch.Tensor, *args: Any, **kwargs: Any) -> Any | torch.Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor. As this plugin only operates
        with a single device, the reduction is simply the identity.

        Args:
            tensor: the tensor to sync and reduce
            *args: ignored
            **kwargs: ignored

        Return:
            the unmodified input as reduction is not needed for single process operation

        """
        return tensor

    @override
    def all_gather(self, tensor: torch.Tensor, group: Any | None=None, sync_grads: bool=False) -> torch.Tensor:
        """Perform a ``all_gather`` on all processes."""
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        pass

    @override
    def broadcast(self, obj: TBroadcast, src: int=0) -> TBroadcast:
        return obj