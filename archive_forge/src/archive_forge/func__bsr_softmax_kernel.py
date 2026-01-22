import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
@triton.jit
def _bsr_softmax_kernel(crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, values_ptr, values_batch_stride, values_row_block_stride, values_nnz_col_block_stride, row_block, col_block, MAX_ROW_NNZ: tl.constexpr, TILE: tl.constexpr):
    batch_pid = tl.program_id(axis=2)
    row_block_offset_pid = tl.program_id(axis=1)
    row_block_pid = tl.program_id(axis=0)
    crow_indices_offset_ptr = crow_indices_ptr + crow_indices_batch_stride * batch_pid + crow_indices_stride * row_block_pid
    nnz_offset = tl.load(crow_indices_offset_ptr)
    nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)
    row_nnz = nnz_offset_next - nnz_offset
    if row_nnz == 0:
        return
    row_arange = tl.arange(0, TILE)
    mask = row_arange < row_nnz * col_block
    curr_row_values_ptrs = values_ptr + values_batch_stride * batch_pid + values_row_block_stride * row_block_offset_pid + nnz_offset * col_block
    row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
    max_row_value = tl.max(row_tile, axis=0)
    for _ in range(TILE, MAX_ROW_NNZ, TILE):
        row_arange += TILE
        mask = row_arange < row_nnz * col_block
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
        curr_max_row_value = tl.max(row_tile, axis=0)
        max_row_value = tl.where(max_row_value > curr_max_row_value, max_row_value, curr_max_row_value)
    num = tl.exp(row_tile - max_row_value)
    denom = tl.sum(num, axis=0)
    for _ in range(TILE, MAX_ROW_NNZ, TILE):
        row_arange -= TILE
        mask = row_arange < row_nnz * col_block
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
        num = tl.exp(row_tile - max_row_value)
        denom += tl.sum(num, axis=0)
    tl.store(curr_row_values_ptrs + row_arange, (num / denom).to(values_ptr.dtype.element_ty), mask=mask)
    for _ in range(TILE, MAX_ROW_NNZ, TILE):
        row_arange += TILE
        mask = row_arange < row_nnz * col_block
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
        num = tl.exp(row_tile - max_row_value)
        tl.store(curr_row_values_ptrs + row_arange, (num / denom).to(values_ptr.dtype.element_ty), mask=mask)