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
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_deformable_detr import DeformableDetrConfig
class DeformableDetrConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by DeformableDetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.use_timm_backbone:
            requires_backends(self, ['timm'])
            kwargs = {}
            if config.dilation:
                kwargs['output_stride'] = 16
            backbone = create_model(config.backbone, pretrained=config.use_pretrained_backbone, features_only=True, out_indices=(2, 3, 4) if config.num_feature_levels > 1 else (4,), in_chans=config.num_channels, **kwargs)
        else:
            backbone = load_backbone(config)
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if 'resnet' in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if 'layer2' not in name and 'layer3' not in name and ('layer4' not in name):
                        parameter.requires_grad_(False)
                elif 'stage.1' not in name and 'stage.2' not in name and ('stage.3' not in name):
                    parameter.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps
        out = []
        for feature_map in features:
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out