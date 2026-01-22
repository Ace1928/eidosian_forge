from itertools import product
import cupy
from cupy._core.internal import _normalize_axis_index
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx.scipy.signal._arraytools import axis_slice
def apply_iir(x, a, axis=-1, zi=None, dtype=None, block_sz=1024):
    if dtype is None:
        dtype = cupy.result_type(x.dtype, a.dtype)
    a = a.astype(dtype)
    if zi is not None:
        zi = zi.astype(dtype)
    x_shape = x.shape
    x_ndim = x.ndim
    axis = _normalize_axis_index(axis, x_ndim)
    k = a.size
    n = x_shape[axis]
    if x_ndim > 1:
        x, x_shape = collapse_2d(x, axis)
        if zi is not None:
            zi, _ = collapse_2d(zi, axis)
    out = cupy.array(x, dtype=dtype, copy=True)
    num_rows = 1 if x.ndim == 1 else x.shape[0]
    n_blocks = (n + block_sz - 1) // block_sz
    total_blocks = num_rows * n_blocks
    correction = cupy.eye(k, dtype=dtype)
    correction = cupy.c_[correction[::-1], cupy.empty((k, block_sz), dtype=dtype)]
    carries = cupy.empty((num_rows, n_blocks, k), dtype=dtype)
    corr_kernel = _get_module_func(IIR_MODULE, 'compute_correction_factors', correction, a)
    first_pass_kernel = _get_module_func(IIR_MODULE, 'first_pass_iir', out)
    second_pass_kernel = _get_module_func(IIR_MODULE, 'second_pass_iir', out)
    carry_correction_kernel = _get_module_func(IIR_MODULE, 'correct_carries', out)
    corr_kernel((k,), (1,), (block_sz, k, a, correction))
    first_pass_kernel((total_blocks,), (block_sz // 2,), (block_sz, k, n, n_blocks, n_blocks * k, correction, out, carries))
    if zi is not None:
        if zi.ndim == 1:
            zi = cupy.broadcast_to(zi, (num_rows, 1, zi.size))
        elif zi.ndim == 2:
            zi = zi.reshape(num_rows, 1, zi.shape[-1])
        if carries.size == 0:
            carries = zi
        else:
            carries = cupy.concatenate((zi, carries), axis=1)
        if not carries.flags.c_contiguous:
            carries = carries.copy()
    if n_blocks > 1 or zi is not None:
        starting_group = int(zi is None)
        blocks_to_merge = n_blocks - starting_group
        carries_stride = (n_blocks + (1 - starting_group)) * k
        carry_correction_kernel((num_rows,), (k,), (block_sz, k, n_blocks, carries_stride, starting_group, correction, carries))
        second_pass_kernel((num_rows * blocks_to_merge,), (block_sz,), (block_sz, k, n, carries_stride, blocks_to_merge, starting_group, correction, carries, out))
    if x_ndim > 1:
        out = out.reshape(x_shape)
        out = cupy.moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()
    return out