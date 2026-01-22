import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
def _softmax(a: torch.Tensor, causal: bool=False) -> torch.Tensor:
    if _has_cpp_library and isinstance(a, SparseCS):
        return a.softmax()
    if a.is_sparse:
        return torch.sparse.softmax(a, dim=a.ndim - 1)
    if _is_triton_available():
        return triton_softmax(a, mask=None, causal=causal)
    else:
        return torch.softmax(a, dim=a.ndim - 1)