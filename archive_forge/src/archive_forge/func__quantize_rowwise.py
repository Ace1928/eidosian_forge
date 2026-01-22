import math
import torch
from bitsandbytes.triton.triton_utils import is_triton_available
@triton.autotune(configs=[triton.Config({}, num_stages=1, num_warps=8), triton.Config({}, num_stages=2, num_warps=8), triton.Config({}, num_stages=4, num_warps=8), triton.Config({}, num_stages=8, num_warps=8), triton.Config({}, num_stages=1), triton.Config({}, num_stages=2), triton.Config({}, num_stages=4), triton.Config({}, num_stages=8), triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['n_elements'])
@triton.jit
def _quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements, BLOCK_SIZE: tl.constexpr, P2: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    output = tl.libdevice.llrint(127.0 * (x / max_val))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)