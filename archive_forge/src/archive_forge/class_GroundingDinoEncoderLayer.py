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
class GroundingDinoEncoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.text_enhancer_layer = GroundingDinoTextEnhancerLayer(config)
        self.fusion_layer = GroundingDinoFusionLayer(config)
        self.deformable_layer = GroundingDinoDeformableLayer(config)

    def get_text_position_embeddings(self, text_features: Tensor, text_position_embedding: Optional[torch.Tensor], text_position_ids: Optional[torch.Tensor]) -> Tensor:
        batch_size, seq_length, _ = text_features.shape
        if text_position_embedding is None and text_position_ids is None:
            text_position_embedding = torch.arange(seq_length, device=text_features.device)
            text_position_embedding = text_position_embedding.float()
            text_position_embedding = text_position_embedding.unsqueeze(0).unsqueeze(-1)
            text_position_embedding = text_position_embedding.repeat(batch_size, 1, 1)
            text_position_embedding = get_sine_pos_embed(text_position_embedding, num_pos_feats=self.d_model, exchange_xy=False)
        if text_position_ids is not None:
            text_position_embedding = get_sine_pos_embed(text_position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False)
        return text_position_embedding

    def forward(self, vision_features: Tensor, vision_position_embedding: Tensor, spatial_shapes: Tensor, level_start_index: Tensor, key_padding_mask: Tensor, reference_points: Tensor, text_features: Optional[Tensor]=None, text_attention_mask: Optional[Tensor]=None, text_position_embedding: Optional[Tensor]=None, text_self_attention_masks: Optional[Tensor]=None, text_position_ids: Optional[Tensor]=None):
        text_position_embedding = self.get_text_position_embeddings(text_features, text_position_embedding, text_position_ids)
        (vision_features, vision_fused_attn), (text_features, text_fused_attn) = self.fusion_layer(vision_features=vision_features, text_features=text_features, attention_mask_vision=key_padding_mask, attention_mask_text=text_attention_mask)
        text_features, text_enhanced_attn = self.text_enhancer_layer(hidden_states=text_features, attention_masks=~text_self_attention_masks, position_embeddings=text_position_embedding if text_position_embedding is not None else None)
        vision_features, vision_deformable_attn = self.deformable_layer(hidden_states=vision_features, attention_mask=~key_padding_mask, position_embeddings=vision_position_embedding, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index)
        return ((vision_features, text_features), (vision_fused_attn, text_fused_attn, text_enhanced_attn, vision_deformable_attn))