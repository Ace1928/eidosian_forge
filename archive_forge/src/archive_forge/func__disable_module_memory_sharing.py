import itertools
import os
from dataclasses import dataclass
from multiprocessing.queues import SimpleQueue
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional
import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
from lightning_utilities import apply_to_collection
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.accelerators.cpu import CPUAccelerator
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.distributed import _set_num_threads_if_needed
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
def _disable_module_memory_sharing(data: Any) -> Any:
    """Disables memory sharing on parameters and buffers of `nn.Module`s contained in the given collection.

    Note: This is only required when running on CPU.

    """

    @torch.no_grad()
    def unshare(module: Module) -> Module:
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            tensor.data = tensor.data.clone()
        return module
    return apply_to_collection(data, function=unshare, dtype=Module)