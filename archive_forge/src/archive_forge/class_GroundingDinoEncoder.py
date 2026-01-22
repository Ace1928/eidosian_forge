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
class GroundingDinoEncoder(GroundingDinoPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`GroundingDinoEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layers = nn.ModuleList([GroundingDinoEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device), torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, vision_features: Tensor, vision_attention_mask: Tensor, vision_position_embedding: Tensor, spatial_shapes: Tensor, level_start_index: Tensor, valid_ratios=None, text_features: Optional[Tensor]=None, text_attention_mask: Optional[Tensor]=None, text_position_embedding: Optional[Tensor]=None, text_self_attention_masks: Optional[Tensor]=None, text_position_ids: Optional[Tensor]=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            vision_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 0 for pixel features that are real (i.e. **not masked**),
                - 1 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            vision_position_embedding (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Flattened text features that are passed to the encoder.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            text_position_embedding (`torch.FloatTensor` of shape `(batch_size, text_seq_len)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            text_self_attention_masks (`torch.BoolTensor` of shape `(batch_size, text_seq_len, text_seq_len)`):
                Masks to avoid performing attention between padding text features. Mask values selected in `[0, 1]`:
                - 1 for text features that are real (i.e. **not masked**),
                - 0 for text features that are padding (i.e. **masked**).
            text_position_ids (`torch.LongTensor` of shape `(batch_size, num_queries)`):
                Position ids for text features.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=vision_features.device)
        encoder_vision_states = () if output_hidden_states else None
        encoder_text_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        all_attn_fused_text = () if output_attentions else None
        all_attn_fused_vision = () if output_attentions else None
        all_attn_enhanced_text = () if output_attentions else None
        all_attn_deformable = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_vision_states += (vision_features,)
                encoder_text_states += (text_features,)
            (vision_features, text_features), attentions = encoder_layer(vision_features=vision_features, vision_position_embedding=vision_position_embedding, spatial_shapes=spatial_shapes, level_start_index=level_start_index, key_padding_mask=vision_attention_mask, reference_points=reference_points, text_features=text_features, text_attention_mask=text_attention_mask, text_position_embedding=text_position_embedding, text_self_attention_masks=text_self_attention_masks, text_position_ids=text_position_ids)
            if output_attentions:
                all_attn_fused_vision += (attentions[0],)
                all_attn_fused_text += (attentions[1],)
                all_attn_enhanced_text += (attentions[2],)
                all_attn_deformable += (attentions[3],)
        if output_hidden_states:
            encoder_vision_states += (vision_features,)
            encoder_text_states += (text_features,)
        if output_attentions:
            all_attns = (all_attn_fused_vision, all_attn_fused_text, all_attn_enhanced_text, all_attn_deformable)
        if not return_dict:
            enc_outputs = [vision_features, text_features, encoder_vision_states, encoder_text_states, all_attns]
            return tuple((v for v in enc_outputs if v is not None))
        return GroundingDinoEncoderOutput(last_hidden_state_vision=vision_features, last_hidden_state_text=text_features, vision_hidden_states=encoder_vision_states, text_hidden_states=encoder_text_states, attentions=all_attns)