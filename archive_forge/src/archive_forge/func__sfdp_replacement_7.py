import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_replacement_7(query, key, value, dropout_p):
    counters['inductor']['fuse_attention'] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False)