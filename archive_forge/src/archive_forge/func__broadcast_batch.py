import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
def _broadcast_batch(mask, batch_size):
    if mask.ndim == 3:
        return mask
    assert mask.ndim == 2
    mask = mask.coalesce()
    values = mask.values()
    indices = mask.indices()
    nnz = len(values)
    indices = indices.repeat(1, batch_size)
    batch_indices = torch.arange(batch_size, device=indices.device)
    batch_indices = batch_indices[:, None].expand(batch_size, nnz).flatten()
    indices = torch.cat([batch_indices[None, :], indices], dim=0)
    values = values.repeat(batch_size)
    size = (batch_size,) + mask.shape
    return torch.sparse_coo_tensor(indices, values, size)