from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def wrap_tensor(x, scalar_ty):
    res_ty = tl.block_type(scalar_ty, shape)
    return tl.tensor(x, res_ty)