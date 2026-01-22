import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.init import constant_
from xformers.components.attention import Attention
from xformers.components.input_projection import InputProjection, InputProjectionConfig
from xformers.components.positional_embedding import RotaryEmbedding
@dataclass
class MultiHeadDispatchConfig:
    dim_model: int
    num_heads: int
    attention: Attention
    bias: bool
    residual_dropout: float
    dim_key: Optional[int]
    dim_value: Optional[int]
    in_proj_container: Optional[InputProjection]
    use_separate_proj_weight: Optional[bool]
    use_rotary_embeddings: Optional[bool]
    out_proj: Optional[nn.Module]

    def __getitem__(self, item):
        return getattr(self, item)