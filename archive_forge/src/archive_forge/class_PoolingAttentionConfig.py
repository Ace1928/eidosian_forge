import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
@dataclass
class PoolingAttentionConfig(AttentionConfig):
    pool_size: int
    stride: Optional[int]
    padding: Optional[int]