import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.utils.distributed import get_dim_for_local_rank
def _update_kv_cache(self, kv, inference_params):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    assert self.layer_idx is not None, 'Generation requires layer_idx in the constructor'
    return _update_kv_cache(kv, inference_params, self.layer_idx)