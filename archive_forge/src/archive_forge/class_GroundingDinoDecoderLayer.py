import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig
class GroundingDinoDecoderLayer(nn.Module):

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = GroundingDinoMultiheadAttention(config, num_attention_heads=config.decoder_attention_heads)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.encoder_attn_text = GroundingDinoMultiheadAttention(config, num_attention_heads=config.decoder_attention_heads)
        self.encoder_attn_text_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.encoder_attn = GroundingDinoMultiscaleDeformableAttention(config, num_heads=config.decoder_attention_heads, n_points=config.decoder_n_points)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Optional[torch.Tensor]=None, reference_points=None, spatial_shapes=None, level_start_index=None, vision_encoder_hidden_states: Optional[torch.Tensor]=None, vision_encoder_attention_mask: Optional[torch.Tensor]=None, text_encoder_hidden_states: Optional[torch.Tensor]=None, text_encoder_attention_mask: Optional[torch.Tensor]=None, self_attn_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False):
        residual = hidden_states
        queries = keys = self.with_pos_embed(hidden_states, position_embeddings)
        hidden_states, self_attn_weights = self.self_attn(queries=queries, keys=keys, values=hidden_states, attention_mask=self_attn_mask, output_attentions=True)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        second_residual = hidden_states
        queries = self.with_pos_embed(hidden_states, position_embeddings)
        hidden_states, text_cross_attn_weights = self.encoder_attn_text(queries=queries, keys=text_encoder_hidden_states, values=text_encoder_hidden_states, output_attentions=True)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_text_layer_norm(hidden_states)
        third_residual = hidden_states
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(hidden_states=hidden_states, attention_mask=vision_encoder_attention_mask, encoder_hidden_states=vision_encoder_hidden_states, encoder_attention_mask=vision_encoder_attention_mask, position_embeddings=position_embeddings, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = third_residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, text_cross_attn_weights, cross_attn_weights)
        return outputs