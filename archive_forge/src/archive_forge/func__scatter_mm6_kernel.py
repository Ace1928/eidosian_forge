import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
@triton.jit
def _scatter_mm6_kernel(nbatches, Ms, Ks: tl.constexpr, N, blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K, others_ptr, others_stride_B, others_stride_K, others_stride_N, accumulators_ptr, accumulators_stride_B, accumulators_stride_M, accumulators_stride_N, c_indices_ptr, r_offsets_ptr, p_offsets_ptr, q_offsets_ptr, is_compressed: tl.constexpr, dot_out_dtype: tl.constexpr, SPLIT_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, GROUP_SIZE: tl.constexpr, allow_tf32: tl.constexpr):
    Ns = N // SPLIT_N
    BLOCKS_M = Ms // TILE_M
    BLOCKS_N = Ns // TILE_N
    pid_t_ = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_b = pid_t_ % nbatches
    pid_t = pid_t_ // nbatches
    num_pid_in_group = GROUP_SIZE * BLOCKS_N
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(BLOCKS_M - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    rm = pid_m * TILE_M + tl.arange(0, TILE_M)
    rn = pid_n * TILE_N + tl.arange(0, TILE_N)
    rk = tl.arange(0, Ks)
    A_ptr = blocks_ptr + (rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K)
    B_ptr = others_ptr + pid_b * others_stride_B + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)
    r = tl.load(r_offsets_ptr + pid_t)
    if is_compressed:
        m = r // N // Ms
        n = r % N // Ns
        r0 = tl.load(c_indices_ptr + m)
        r1 = tl.load(c_indices_ptr + m + 1)
        g0 = n * r1 + (SPLIT_N - n) * r0
        nnz = r1 - r0
    else:
        g0 = tl.load(c_indices_ptr + pid_t)
        g1 = tl.load(c_indices_ptr + pid_t + 1)
        nnz = g1 - g0
    q_ptr = q_offsets_ptr + g0
    acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)
    if is_compressed:
        A_ptr += r0 * blocks_stride_P
        for _ in range(nnz):
            q = tl.load(q_ptr)
            B = tl.load(B_ptr + q)
            A = tl.load(A_ptr)
            acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
            A_ptr += blocks_stride_P
            q_ptr += 1
    else:
        p_ptr = p_offsets_ptr + g0
        for _ in range(nnz):
            q = tl.load(q_ptr)
            B = tl.load(B_ptr + q)
            p = tl.load(p_ptr)
            A = tl.load(A_ptr + p * blocks_stride_P)
            p_ptr += 1
            q_ptr += 1
            acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
    C_ptr = accumulators_ptr + r + pid_b * accumulators_stride_B + (rm[:, None] * accumulators_stride_M + rn[None, :] * accumulators_stride_N)
    tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))