from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _store_legacy(ptr, val, mask, boundary_check, cache, eviction, builder):
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f'Unsupported ptr type {ptr.type.__repr__()} in `tl.store`')
    if boundary_check:
        raise ValueError('`boundary_check` argument is not supported for storing a tensor of pointers or storing a scalar. Because the compiler does not know the boundary; please use block pointers (defined by `make_block_ptr`) instead')
    if not ptr.type.is_block():
        if val.type.is_block():
            raise ValueError('Value argument cannot be block type if pointer argument is not a block')
        if mask and mask.type.is_block():
            raise ValueError('Mask argument cannot be block type if pointer argument is not a block')
    if ptr.type.is_block():
        val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
        if mask:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)
    val = cast(val, elt_ty, builder)
    if not mask:
        return tl.tensor(builder.create_store(ptr.handle, val.handle, cache, eviction), tl.void)
    if not mask.type.scalar.is_bool():
        raise ValueError('Mask must have boolean scalar type')
    return tl.tensor(builder.create_masked_store(ptr.handle, val.handle, mask.handle, cache, eviction), tl.void)