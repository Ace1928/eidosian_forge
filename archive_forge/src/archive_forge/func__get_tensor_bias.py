from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
def _get_tensor_bias(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> Optional[torch.Tensor]:
    if isinstance(attn_bias, torch.Tensor):
        return attn_bias
    elif isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        return attn_bias._bias
    return None