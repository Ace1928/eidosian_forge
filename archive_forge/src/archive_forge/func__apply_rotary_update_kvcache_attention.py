import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.utils.distributed import get_dim_for_local_rank
def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
    """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
    assert inference_params is not None and inference_params.seqlen_offset > 0
    assert self.use_flash_attn
    if self.rotary_emb_dim > 0:
        assert self.rotary_emb.scale is None, 'This code path does not support xPos'
        self.rotary_emb._update_cos_sin_cache(inference_params.max_seqlen, device=q.device, dtype=q.dtype)
        rotary_cos, rotary_sin = (self.rotary_emb._cos_cached, self.rotary_emb._sin_cached)
    else:
        rotary_cos, rotary_sin = (None, None)
    batch = q.shape[0]
    kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
    cache_seqlens = inference_params.lengths_per_sample[:batch] if inference_params.lengths_per_sample is not None else inference_params.seqlen_offset
    alibi_slopes = getattr(self.inner_cross_attn, 'alibi_slopes', None)
    context = flash_attn_with_kvcache(q, kv_cache[:, :, 0], kv_cache[:, :, 1], kv[:, :, 0], kv[:, :, 1], rotary_cos=rotary_cos, rotary_sin=rotary_sin, cache_seqlens=cache_seqlens, softmax_scale=self.inner_cross_attn.softmax_scale, causal=self.inner_cross_attn.causal, rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False, alibi_slopes=alibi_slopes)
    return context