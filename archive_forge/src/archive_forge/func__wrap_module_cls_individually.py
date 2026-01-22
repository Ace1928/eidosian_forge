import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def _wrap_module_cls_individually(module: nn.Module, module_classes: Sequence[type], recurse: bool, *args, **kwargs):
    if recurse:
        return True
    else:
        return isinstance(module, tuple(module_classes))