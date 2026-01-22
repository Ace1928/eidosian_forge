import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
@triton.jit
def _scatter_mm2_kernel(M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K, others_ptr, others_stride_Q, others_stride_K, others_stride_N, accumulators_ptr, accumulators_stride_R, accumulators_stride_M, accumulators_stride_N, pq_offsets_ptr, pq_offsets_stride, pq_ptr, pq_stride_T, pq_stride_1, dot_out_dtype: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, allow_tf32: tl.constexpr):
    Ms = M // TILE_M
    Ns = N // TILE_N
    pid_t = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // Ms
    pid_n = pid % Ms
    rm = pid_m * TILE_M + tl.arange(0, TILE_M)
    rn = pid_n * TILE_N + tl.arange(0, TILE_N)
    rk = tl.arange(0, K)
    A_ptr = blocks_ptr + (rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K)
    B_ptr = others_ptr + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)
    g0 = tl.load(pq_offsets_ptr + pid_t * pq_offsets_stride)
    g1 = tl.load(pq_offsets_ptr + (pid_t + 1) * pq_offsets_stride)
    if g0 == g1:
        return
    acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)
    for i in range(g0, g1):
        p = tl.load(pq_ptr + i * pq_stride_T)
        q = tl.load(pq_ptr + i * pq_stride_T + pq_stride_1)
        A = tl.load(A_ptr + p * blocks_stride_P)
        B = tl.load(B_ptr + q * others_stride_Q)
        acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
    C_ptr = accumulators_ptr + pid_t * accumulators_stride_R + (rm[:, None] * accumulators_stride_M + rn[None, :] * accumulators_stride_N)
    tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))