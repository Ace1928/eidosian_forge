import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
def type_before_parametrizations(module: Module) -> type:
    """Return the module type before parametrizations were applied and if not, then it returns the module type.

    Args:
        module (nn.Module): module to get type of
    """
    if is_parametrized(module):
        return module.__class__.__bases__[0]
    else:
        return type(module)