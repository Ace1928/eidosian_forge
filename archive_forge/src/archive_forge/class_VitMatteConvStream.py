from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig
class VitMatteConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """

    def __init__(self, config):
        super().__init__()
        in_channels = config.backbone_config.num_channels
        out_channels = config.convstream_hidden_sizes
        self.convs = nn.ModuleList()
        self.conv_chans = [in_channels] + out_channels
        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            self.convs.append(VitMatteBasicConv3x3(config, in_chan_, out_chan_))

    def forward(self, pixel_values):
        out_dict = {'detailed_feature_map_0': pixel_values}
        embeddings = pixel_values
        for i in range(len(self.convs)):
            embeddings = self.convs[i](embeddings)
            name_ = 'detailed_feature_map_' + str(i + 1)
            out_dict[name_] = embeddings
        return out_dict