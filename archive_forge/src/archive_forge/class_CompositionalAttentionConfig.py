import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from xformers.components.attention import (
from xformers.components.attention.core import _softmax
from xformers.components.input_projection import InputProjection, InputProjectionConfig
@dataclass
class CompositionalAttentionConfig(AttentionConfig):
    dim_model: int
    num_heads: int
    dim_attn: Optional[int] = None
    num_rules: Optional[int] = None
    dim_key: Optional[int] = None
    dim_value: Optional[int] = None
    dim_selection: Optional[int] = None
    dropout: float
    qk_rule: bool = False
    nonlinear: bool = False
    q_compose: bool = False
    bias: bool = True
    causal: Optional[bool] = False
    in_proj_container: Optional[InputProjection] = None
    use_separate_proj_weight: Optional[bool] = False