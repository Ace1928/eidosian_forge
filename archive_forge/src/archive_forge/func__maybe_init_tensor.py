import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def _maybe_init_tensor(module: nn.Module, attr: str, distribution_: Callable, **kwargs):
    if hasattr(module, attr):
        maybe_tensor = getattr(module, attr)
        if maybe_tensor is not None and isinstance(maybe_tensor, torch.Tensor):
            distribution_(maybe_tensor, **kwargs)