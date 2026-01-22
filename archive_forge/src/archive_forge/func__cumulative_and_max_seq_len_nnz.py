import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _cumulative_and_max_seq_len_nnz(qkv: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if not isinstance(qkv, NestedTensor):
        raise ValueError('QKV must be nested for flash cumulative_seq_len calculation.')
    if qkv.lengths() is None:
        cumulative_seqlen = qkv.offsets().to(dtype=torch.int32, device=qkv.device)
        max_seqlen = qkv._max_seqlen
        n_elem = qkv.values().shape[0]
    else:
        cumulative_seqlen = qkv.lengths().cumsum(0).to(dtype=torch.int32, device=qkv.device)
        batch_size = qkv.size(0)
        max_seqlen = qkv._max_seqlen
        n_elem = int(cumulative_seqlen[-1].item())
    return (cumulative_seqlen, max_seqlen, n_elem)