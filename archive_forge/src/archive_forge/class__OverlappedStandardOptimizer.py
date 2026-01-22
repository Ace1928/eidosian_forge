from abc import ABC, abstractmethod
import inspect
from typing import Dict, Type
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.distributed.optim import as_functional_optim
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
@register_overlapped(Optimizer)
class _OverlappedStandardOptimizer(OverlappedOptimizer):
    """Overlaps a regular ``Optimizer``."""

    def __init__(self, optim_cls: Type, params, *optim_args, **optim_kwargs) -> None:
        super().__init__(optim_cls)
        f_optim = as_functional_optim(self.optim_cls, *optim_args, **optim_kwargs)
        self._opt_hook_state = _OptimizerHookState(f_optim, params)

    def register_ddp(self, ddp_inst: DistributedDataParallel):
        ddp_inst.register_comm_hook(None, _hook_then_optimizer(allreduce_hook, self._opt_hook_state))

    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Register the overlapped optimizer with FSDP."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support overlapped FSDP.')