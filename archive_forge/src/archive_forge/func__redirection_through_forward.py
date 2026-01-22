import inspect
from copy import deepcopy
from functools import wraps
from typing import (
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch import nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override
from lightning_fabric.plugins import Precision
from lightning_fabric.strategies import Strategy
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.data import _set_sampler_epoch
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import Optimizable
def _redirection_through_forward(self, method_name: str) -> Callable:
    assert method_name != 'forward'
    original_forward = self._original_module.forward

    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        self._original_module.forward = original_forward
        method = getattr(self._original_module, method_name)
        return method(*args, **kwargs)

    def call_forward_module(*args: Any, **kwargs: Any) -> Any:
        self._original_module.forward = wrapped_forward
        return self.forward(*args, **kwargs)
    return call_forward_module