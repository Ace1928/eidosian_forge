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
class OneFormerTextTransformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None, use_checkpoint=False, layer_norm_eps=1e-05):
        super().__init__()
        self.width = width
        self.num_layers = layers
        self.layers = nn.Sequential(*[OneFormerTextTransformerLayer(width, heads, attn_mask, layer_norm_eps) for _ in range(layers)])
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states: torch.Tensor):
        for layer in self.layers:
            if self.use_checkpoint:
                hidden_states = self._gradient_checkpointing_func(layer, hidden_states)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states