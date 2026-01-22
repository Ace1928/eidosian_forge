import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_glpn import GLPNConfig
class GLPNSelectiveFeatureFusion(nn.Module):
    """
    Selective Feature Fusion module, as explained in the [paper](https://arxiv.org/abs/2201.07436) (section 3.4). This
    module adaptively selects and integrates local and global features by attaining an attention map for each feature.
    """

    def __init__(self, in_channel=64):
        super().__init__()
        self.convolutional_layer1 = nn.Sequential(nn.Conv2d(in_channels=int(in_channel * 2), out_channels=in_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU())
        self.convolutional_layer2 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel / 2), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(in_channel / 2)), nn.ReLU())
        self.convolutional_layer3 = nn.Conv2d(in_channels=int(in_channel / 2), out_channels=2, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, local_features, global_features):
        features = torch.cat((local_features, global_features), dim=1)
        features = self.convolutional_layer1(features)
        features = self.convolutional_layer2(features)
        features = self.convolutional_layer3(features)
        attn = self.sigmoid(features)
        hybrid_features = local_features * attn[:, 0, :, :].unsqueeze(1) + global_features * attn[:, 1, :, :].unsqueeze(1)
        return hybrid_features