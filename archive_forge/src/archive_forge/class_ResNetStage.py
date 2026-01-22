from typing import Optional
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_resnet import ResNetConfig
class ResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    def __init__(self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int=2, depth: int=2):
        super().__init__()
        layer = ResNetBottleNeckLayer if config.layer_type == 'bottleneck' else ResNetBasicLayer
        if config.layer_type == 'bottleneck':
            first_layer = layer(in_channels, out_channels, stride=stride, activation=config.hidden_act, downsample_in_bottleneck=config.downsample_in_bottleneck)
        else:
            first_layer = layer(in_channels, out_channels, stride=stride, activation=config.hidden_act)
        self.layers = nn.Sequential(first_layer, *[layer(out_channels, out_channels, activation=config.hidden_act) for _ in range(depth - 1)])

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state