import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.feature_maps import (
@staticmethod
def _causal_attention(k_prime: torch.Tensor, q_prime: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_v = torch.ones_like(v.unsqueeze(2))
    Gps = k_prime.unsqueeze(3) * v.unsqueeze(2)
    Grenorm = k_prime.unsqueeze(3) * ref_v
    att_raw = torch.einsum('bcfe,bcf->bce', Gps, q_prime)
    att_norm = torch.einsum('bcfe,bcf->bce', Grenorm, q_prime)
    att_raw = att_raw.cumsum(2)
    att_norm = att_norm.cumsum(2)
    return (att_raw, att_norm)