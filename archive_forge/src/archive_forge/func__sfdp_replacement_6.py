import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_replacement_6(query, key, value, attn_mask, dropout_p):
    counters['inductor']['fuse_attention'] += 1
    return aten.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=attn_mask.to(dtype=query.dtype), dropout_p=dropout_p, is_causal=False)