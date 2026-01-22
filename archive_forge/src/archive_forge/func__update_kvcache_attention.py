import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.utils.distributed import get_dim_for_local_rank
def _update_kvcache_attention(self, q, kv, inference_params):
    """Write kv to inference_params, then do attention"""
    if inference_params.seqlen_offset == 0 or not self.use_flash_attn:
        kv = self._update_kv_cache(kv, inference_params)
        return self.inner_cross_attn(q, kv)
    else:
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = inference_params.lengths_per_sample[:batch] if inference_params.lengths_per_sample is not None else inference_params.seqlen_offset
        alibi_slopes = getattr(self.inner_cross_attn, 'alibi_slopes', None)
        context = flash_attn_with_kvcache(q, kv_cache[:, :, 0], kv_cache[:, :, 1], kv[:, :, 0], kv[:, :, 1], cache_seqlens=cache_seqlens, softmax_scale=self.inner_cross_attn.softmax_scale, causal=self.inner_cross_attn.causal, alibi_slopes=alibi_slopes)
        return context