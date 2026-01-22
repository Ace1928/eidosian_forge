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
class OneFormerPixelLevelModule(nn.Module):

    def __init__(self, config: OneFormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`OneFormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        self.encoder = load_backbone(config)
        self.decoder = OneFormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool=False) -> OneFormerPixelLevelModuleOutput:
        features: List[Tensor] = self.encoder(pixel_values).feature_maps
        decoder_output: OneFormerPixelDecoderOutput = self.decoder(features, output_hidden_states=output_hidden_states)
        return OneFormerPixelLevelModuleOutput(encoder_features=tuple(features), decoder_features=decoder_output.multi_scale_features, decoder_last_feature=decoder_output.mask_features)