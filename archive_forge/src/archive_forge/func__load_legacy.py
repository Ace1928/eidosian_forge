from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f'Unsupported ptr type {ptr.type.__repr__()} in `tl.load`')
    if not mask and other:
        raise ValueError('`other` cannot be provided without `mask`')
    if padding or boundary_check:
        raise ValueError('`padding_option` or `boundary_check` argument is not supported for loading a tensor ofpointers or loading a scalar. Because the compiler does not know the boundary; please use block pointers (defined by `make_block_ptr`) instead')
    if not ptr.type.is_block():
        if mask and mask.type.is_block():
            raise ValueError('Mask argument cannot be block type if pointer argument is not a block')
        if other and other.type.is_block():
            raise ValueError('Other argument cannot be block type if pointer argument is not a block')
    if ptr.type.is_block():
        if mask:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if other:
            other = broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)
    if other:
        other = cast(other, elt_ty, builder)
    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        dst_ty = elt_ty
    if not mask:
        return tl.tensor(builder.create_load(ptr.handle, cache, eviction, is_volatile), dst_ty)
    else:
        return tl.tensor(builder.create_masked_load(ptr.handle, mask.handle, other.handle if other else None, cache, eviction, is_volatile), dst_ty)