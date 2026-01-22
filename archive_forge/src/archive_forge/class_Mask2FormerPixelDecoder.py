import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_accelerate_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_mask2former import Mask2FormerConfig
class Mask2FormerPixelDecoder(nn.Module):

    def __init__(self, config: Mask2FormerConfig, feature_channels):
        super().__init__()
        self.config = config
        feature_dim = config.feature_size
        mask_dim = config.mask_feature_size
        num_pos_features = feature_dim // 2
        self.position_embedding = Mask2FormerSinePositionEmbedding(num_pos_feats=num_pos_features, normalize=True)
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels:]
        self.transformer_feature_strides = config.feature_strides[-self.num_feature_levels:]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, feature_dim))
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(nn.Sequential(nn.Conv2d(in_channels, feature_dim, kernel_size=1), nn.GroupNorm(32, feature_dim)))
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList([nn.Sequential(nn.Conv2d(transformer_in_channels[-1], feature_dim, kernel_size=1), nn.GroupNorm(32, feature_dim))])
        self.encoder = Mask2FormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(feature_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False), nn.GroupNorm(32, feature_dim))
            output_conv = nn.Sequential(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False), nn.GroupNorm(32, feature_dim), nn.ReLU())
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(self, features, encoder_outputs=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        input_embeds = []
        position_embeddings = []
        for level, x in enumerate(features[::-1][:self.num_feature_levels]):
            input_embeds.append(self.input_projections[level](x))
            position_embeddings.append(self.position_embedding(x))
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in input_embeds]
        spatial_shapes = [(embed.shape[2], embed.shape[3]) for embed in input_embeds]
        input_embeds_flat = torch.cat([embed.flatten(2).transpose(1, 2) for embed in input_embeds], 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=input_embeds_flat.device)
        masks_flat = torch.cat([mask.flatten(1) for mask in masks], 1)
        position_embeddings = [embed.flatten(2).transpose(1, 2) for embed in position_embeddings]
        level_pos_embed_flat = [x + self.level_embed[i].view(1, 1, -1) for i, x in enumerate(position_embeddings)]
        level_pos_embed_flat = torch.cat(level_pos_embed_flat, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(mask, dtype=input_embeds_flat.dtype) for mask in masks], 1)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(inputs_embeds=input_embeds_flat, attention_mask=masks_flat, position_embeddings=level_pos_embed_flat, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]
        split_sizes = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_sizes[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_sizes[i] = last_hidden_state.shape[1] - level_start_index[i]
        encoder_output = torch.split(last_hidden_state, [size.item() for size in split_sizes], dim=1)
        outputs = [x.transpose(1, 2).view(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1]) for i, x in enumerate(encoder_output)]
        for idx, feature in enumerate(features[:self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convolutions[idx]
            output_conv = self.output_convolutions[idx]
            current_fpn = lateral_conv(feature)
            out = current_fpn + nn.functional.interpolate(outputs[-1], size=current_fpn.shape[-2:], mode='bilinear', align_corners=False)
            out = output_conv(out)
            outputs.append(out)
        num_cur_levels = 0
        multi_scale_features = []
        for out in outputs:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(out)
                num_cur_levels += 1
        return Mask2FormerPixelDecoderOutput(mask_features=self.mask_projection(outputs[-1]), multi_scale_features=tuple(multi_scale_features), attentions=encoder_outputs.attentions)