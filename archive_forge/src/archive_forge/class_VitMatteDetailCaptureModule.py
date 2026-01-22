from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig
class VitMatteDetailCaptureModule(nn.Module):
    """
    Simple and lightweight Detail Capture Module for ViT Matting.
    """

    def __init__(self, config):
        super().__init__()
        if len(config.fusion_hidden_sizes) != len(config.convstream_hidden_sizes) + 1:
            raise ValueError('The length of fusion_hidden_sizes should be equal to the length of convstream_hidden_sizes + 1.')
        self.config = config
        self.convstream = VitMatteConvStream(config)
        self.conv_chans = self.convstream.conv_chans
        self.fusion_blocks = nn.ModuleList()
        self.fusion_channels = [config.hidden_size] + config.fusion_hidden_sizes
        for i in range(len(self.fusion_channels) - 1):
            self.fusion_blocks.append(VitMatteFusionBlock(config=config, in_channels=self.fusion_channels[i] + self.conv_chans[-(i + 1)], out_channels=self.fusion_channels[i + 1]))
        self.matting_head = VitMatteHead(config)

    def forward(self, features, pixel_values):
        detail_features = self.convstream(pixel_values)
        for i in range(len(self.fusion_blocks)):
            detailed_feature_map_name = 'detailed_feature_map_' + str(len(self.fusion_blocks) - i - 1)
            features = self.fusion_blocks[i](features, detail_features[detailed_feature_map_name])
        alphas = torch.sigmoid(self.matting_head(features))
        return alphas