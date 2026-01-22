import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
class MaskFormerPixelDecoder(nn.Module):

    def __init__(self, *args, feature_size: int=256, mask_feature_size: int=256, **kwargs):
        """
        Pixel Decoder Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It first runs the backbone's features into a Feature Pyramid
        Network creating a list of feature maps. Then, it projects the last one to the correct `mask_size`.

        Args:
            feature_size (`int`, *optional*, defaults to 256):
                The feature size (channel dimension) of the FPN feature maps.
            mask_feature_size (`int`, *optional*, defaults to 256):
                The features (channels) of the target masks size \\\\(C_{\\epsilon}\\\\) in the paper.
        """
        super().__init__()
        self.fpn = MaskFormerFPNModel(*args, feature_size=feature_size, **kwargs)
        self.mask_projection = nn.Conv2d(feature_size, mask_feature_size, kernel_size=3, padding=1)

    def forward(self, features: List[Tensor], output_hidden_states: bool=False, return_dict: bool=True) -> MaskFormerPixelDecoderOutput:
        fpn_features = self.fpn(features)
        last_feature_projected = self.mask_projection(fpn_features[-1])
        if not return_dict:
            return (last_feature_projected, tuple(fpn_features)) if output_hidden_states else (last_feature_projected,)
        return MaskFormerPixelDecoderOutput(last_hidden_state=last_feature_projected, hidden_states=tuple(fpn_features) if output_hidden_states else ())