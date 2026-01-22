from abc import ABC, abstractmethod
import inspect
from typing import Dict, Type
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.distributed.optim import as_functional_optim
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
class OverlappedOptimizer(ABC):

    def __init__(self, optim_cls: Type) -> None:
        """
        Initialize the OverlappedOptimizer.

        Overlappedoptimizer is a base class that child classes can implement to
        specify how different optimizers will register themselves with DDP.
        """
        self.optim_cls = optim_cls

    @abstractmethod
    def register_ddp(self, ddp: DistributedDataParallel) -> None:
        """Registers the overlapped optimizer with DDP."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support overlapped DDP.')

    @abstractmethod
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Registers the overlapped optimizer with FSDP."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support overlapped FSDP.')