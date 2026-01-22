import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerTextContextDecoder(nn.Module):

    def __init__(self, transformer_width=256, transformer_heads=4, transformer_layers=6, visual_dim=1024, dropout=0.1, layer_norm_eps=1e-05, **kwargs):
        super().__init__()
        self.memory_proj = nn.Sequential(nn.LayerNorm(visual_dim, eps=layer_norm_eps), nn.Linear(visual_dim, transformer_width), nn.LayerNorm(transformer_width, eps=layer_norm_eps))
        self.text_proj = nn.Sequential(nn.LayerNorm(visual_dim, eps=layer_norm_eps), nn.Linear(visual_dim, transformer_width))
        self.decoder = nn.ModuleList([OneFormerTextTransformerDecoderLayer(transformer_width, transformer_heads, dropout, layer_norm_eps) for _ in range(transformer_layers)])
        self.out_proj = nn.Sequential(nn.LayerNorm(transformer_width, eps=layer_norm_eps), nn.Linear(transformer_width, visual_dim))

    def forward(self, text, visual):
        visual = self.memory_proj(visual)
        hidden_state = self.text_proj(text)
        for layer in self.decoder:
            hidden_state = layer(hidden_state, visual)
        return self.out_proj(hidden_state)