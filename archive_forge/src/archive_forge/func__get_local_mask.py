from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
from xformers.components.attention import (
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import scaled_dot_product_attention
def _get_local_mask(self, shape: torch.Size) -> torch.Tensor:
    window_size = self.window_size * 2 + 1 if self.causal else self.window_size
    mask = local_1d_pattern(shape[1], window_size)
    if self.causal:
        mask &= causal_1d_pattern(shape[1])
    mask = sparsify(mask) if self.force_sparsity else maybe_sparsify(mask)
    return mask