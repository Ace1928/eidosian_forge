import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def _scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor], dropout_p: float=0.0, is_causal: bool=False, scale: Optional[float]=None):
    f_name = '_scaled_dot_product_attention'
    check(not is_causal, f'{f_name}(): is_causal == True is not supported.')
    check(attn_mask is not None, f'{f_name}(): attn_mask == None is not supported.')
    assert attn_mask is not None
    check(attn_mask.layout == torch.sparse_bsr, f'{f_name}(): attn_mask.layout must be {torch.sparse_bsr}, but got attn_mask.layout == {attn_mask.layout}.')
    check_device(f_name, key, query.device)
    check_device(f_name, value, query.device)
    check_device(f_name, attn_mask, query.device)
    check_dtype(f_name, key, query.dtype)
    check_dtype(f_name, value, query.dtype)
    if attn_mask.dtype is not torch.bool:
        check_dtype(f_name, attn_mask, query.dtype)
    sdpa = sampled_addmm(attn_mask, query, key.transpose(-2, -1), beta=0.0, skip_checks=False)
    if scale is None and query.size(-1) == 0 or scale == 0.0:
        check(False, f'{f_name}(): current value of scale == {scale} results in division by zero.')
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    sdpa.values().mul_(scale_factor)
    sdpa = bsr_softmax(sdpa)
    torch.nn.functional.dropout(sdpa.values(), p=dropout_p, inplace=True)
    sdpa = bsr_dense_mm(sdpa, value)
    return sdpa