import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def _sfdp_pattern_5(query, key, value, attn_mask):
    attn_weight = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + attn_mask, dim=-1)
    return attn_weight @ value