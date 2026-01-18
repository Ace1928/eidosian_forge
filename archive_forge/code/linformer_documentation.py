from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import scaled_dot_product_attention

        Linformer attention mechanism,
        from `Linformer: Self-Attention with Linear Complexity`_, Wang et al (2020).
        The original notation is kept as is.

        .. _`Linformer: Self-Attention with Linear Complexity` : https://arxiv.org/abs/2006.04768v2
        