import functools
import logging
import math
import os
import warnings
from contextlib import ExitStack
from functools import partial
from types import ModuleType
from typing import Any, Callable, ContextManager, Literal, Optional, OrderedDict, Set, Tuple, Type, cast
import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import init
from torch.nn.modules.module import _IncompatibleKeys
from typing_extensions import Self, override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import (
from lightning_fabric.utilities.types import _DEVICE
class _Linear4bit(bnb.nn.Linear4bit):
    """Wraps `bnb.nn.Linear4bit` to enable: instantiation directly on the device, re-quantizaton when loading the
        state dict, meta-device initialization, and materialization."""

    def __init__(self, *args: Any, device: Optional[_DEVICE]=None, **kwargs: Any) -> None:
        super().__init__(*args, device=device, **kwargs)
        self.weight = cast(bnb.nn.Params4bit, self.weight)
        self.bias = cast(Optional[torch.nn.Parameter], self.bias)
        if torch.tensor(0, device=device).device.type == 'cuda':
            self.quantize_()
        self._register_load_state_dict_pre_hook(partial(_quantize_on_load_hook, self.quantize_))
        self.register_load_state_dict_post_hook(_ignore_missing_weights_hook)

    def quantize_(self, weight: Optional[torch.Tensor]=None, device: Optional[torch.device]=None) -> None:
        """Inplace quantize."""
        if weight is None:
            weight = self.weight.data
            if weight.data.type == torch.uint8:
                return
        assert isinstance(self.weight, bnb.nn.Params4bit)
        self.weight = self.quantize(self.weight, weight, device)

    @staticmethod
    def quantize(params4bit: bnb.nn.Params4bit, weight: torch.Tensor, device: Optional[torch.device]) -> bnb.nn.Params4bit:
        device = device or torch.device('cuda')
        if device.type != 'cuda':
            raise RuntimeError(f'Unexpected device type: {device.type}')
        w = weight.contiguous().to(device=device, dtype=torch.half)
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=params4bit.blocksize, compress_statistics=params4bit.compress_statistics, quant_type=params4bit.quant_type)
        return _replace_param(params4bit, w_4bit, quant_state)

    def to_empty(self, *, device: _DEVICE, recurse: bool=True) -> Self:
        if self.weight.dtype == torch.uint8:
            weight = torch.empty(self.weight.quant_state[1], device=device, dtype=torch.half)
        else:
            weight = torch.empty_like(self.weight.data, device=device)
        device = torch.device(device)
        if device.type == 'cuda':
            self.quantize_(weight, device)
        else:
            self.weight = _replace_param(self.weight, weight)
        if self.bias is not None:
            self.bias = _replace_param(self.bias, torch.empty_like(self.bias, device=device))
        return self

    def reset_parameters(self) -> None:
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        linear_init_finished = isinstance(self.weight, bnb.nn.Params4bit)
        if linear_init_finished and self.weight.dtype == torch.uint8:
            weight = torch.empty(self.weight.quant_state[1], device=self.weight.device, dtype=torch.half)
        else:
            weight = self.weight.data
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if linear_init_finished:
            if self.weight.device.type == 'cuda':
                self.quantize_(weight)
            else:
                self.weight = _replace_param(self.weight, weight)