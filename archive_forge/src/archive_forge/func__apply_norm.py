import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _apply_norm(x: TensorLikeType, norm: NormType, signal_numel: int, forward: bool) -> TensorLikeType:
    """Apply normalization to the un-normalized FFT result"""
    torch._check(norm in _NORM_VALUES, lambda: f'Invalid normalization mode: {norm}')
    if norm == 'ortho':
        return x * (1 / math.sqrt(signal_numel))
    normalize = not forward and (norm is None or norm == 'backward') or (forward and norm == 'forward')
    return x * (1 / signal_numel) if normalize else x