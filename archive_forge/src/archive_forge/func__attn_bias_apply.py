import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
def _attn_bias_apply(attn_bias: Optional[Union[torch.Tensor, AttentionBias]], op: Callable[[torch.Tensor], torch.Tensor]) -> Optional[Union[torch.Tensor, AttentionBias]]:
    if isinstance(attn_bias, torch.Tensor):
        return op(attn_bias)
    if isinstance(attn_bias, LowerTriangularMaskWithTensorBias):
        return LowerTriangularMaskWithTensorBias(op(attn_bias._bias))
    return attn_bias