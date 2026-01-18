from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
from typing import Any, TypeVar
import warnings
from ..modules import Module
Remove the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    