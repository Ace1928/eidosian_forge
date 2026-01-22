import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _default_alldims(dim: Optional[DimsType], x: TensorLikeType) -> List[int]:
    """Convert Optional[DimsType] to a simple list, defaulting to all dimensions"""
    if dim is None:
        return list(range(x.ndim))
    elif not isinstance(dim, Sequence):
        return [dim]
    else:
        return list(dim)