from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _load_block_pointer(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    if mask or other:
        raise ValueError('`mask` and `other` arguments cannot be specified for loading block pointers')
    elt_ty = ptr.type.element_ty.element_ty
    assert elt_ty != tl.int1, '`tl.int1` should be rewrited in `tl.make_block_ptr`'
    if elt_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
        raise ValueError('Padding option `nan` is not supported for integer block pointers')
    dst_ty = ptr.type.element_ty
    boundary_check = _canonicalize_boundary_check(boundary_check, dst_ty.get_block_shapes())
    return tl.tensor(builder.create_tensor_pointer_load(ptr.handle, boundary_check, padding, cache, eviction, is_volatile), dst_ty)