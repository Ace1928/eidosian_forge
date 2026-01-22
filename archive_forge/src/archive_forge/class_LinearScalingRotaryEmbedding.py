import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from vllm._C import ops
class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: int, is_neox_style: bool, scaling_factor: float) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(max_len, dtype=torch.float)
        t = t / self.scaling_factor
        freqs = torch.einsum('i,j -> ij', t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache