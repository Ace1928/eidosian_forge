import copy
from typing import Optional, Any, Union, Callable
import torch
import warnings
from torch import Tensor
from .. import functional as F
from .module import Module
from .activation import MultiheadAttention
from .container import ModuleList
from ..init import xavier_uniform_
from .dropout import Dropout
from .linear import Linear
from .normalization import LayerNorm
def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}')