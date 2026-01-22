from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig
class VitMatteHead(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """

    def __init__(self, config):
        super().__init__()
        in_channels = config.fusion_hidden_sizes[-1]
        mid_channels = 16
        self.matting_convs = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(True), nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, hidden_state):
        hidden_state = self.matting_convs(hidden_state)
        return hidden_state