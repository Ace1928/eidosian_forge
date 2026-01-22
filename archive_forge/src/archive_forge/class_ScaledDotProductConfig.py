import logging
from dataclasses import dataclass
from typing import Optional, Union
import torch
from torch import nn
from xformers.components.attention import (
from xformers.components.attention.core import scaled_dot_product_attention
@dataclass
class ScaledDotProductConfig(AttentionConfig):
    causal: Optional[bool]
    seq_len: Optional[int]
    to_seq_len: Optional[int]