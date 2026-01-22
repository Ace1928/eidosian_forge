from dataclasses import dataclass
import torch
from xformers.components.attention import Attention, AttentionConfig, register_attention
@dataclass
class LambdaLayerConfig(AttentionConfig):
    seq_len: int
    dim_head: int