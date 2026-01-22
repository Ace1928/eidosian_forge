import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig
class Kosmos2ImageToTextProjection(nn.Module):
    """The layer that transforms the image model's output to part of the text model's input (namely, image features)"""

    def __init__(self, config: Kosmos2Config):
        super().__init__()
        self.dense = nn.Linear(config.vision_config.hidden_size, config.text_config.embed_dim)
        self.latent_query = nn.Parameter(torch.randn(config.latent_query_num, config.text_config.embed_dim))
        self.x_attn = KosmosTextAttention(config.text_config, config.text_config.embed_dim, config.text_config.attention_heads, dropout=config.text_config.attention_dropout, is_decoder=False, add_inner_attn_layernorm=False)

    def forward(self, features):
        hidden_states = self.dense(features)
        latent_query = self.latent_query.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        key_value_states = torch.cat([hidden_states, latent_query], dim=1)
        hidden_states, attn_weights, _ = self.x_attn(hidden_states=latent_query, encoder_hidden_states=key_value_states, past_key_value=None, attention_mask=None, output_attentions=None)
        return (hidden_states, attn_weights)