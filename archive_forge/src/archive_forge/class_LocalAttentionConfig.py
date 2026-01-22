from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
from xformers.components.attention import (
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import scaled_dot_product_attention
@dataclass
class LocalAttentionConfig(AttentionConfig):
    causal: Optional[bool] = None
    window_size: Optional[int] = None
    force_sparsity: Optional[bool] = None