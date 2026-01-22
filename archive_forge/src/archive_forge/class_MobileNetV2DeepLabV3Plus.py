from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilenet_v2 import MobileNetV2Config
class MobileNetV2DeepLabV3Plus(nn.Module):
    """
    The neural network from the paper "Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation" https://arxiv.org/abs/1802.02611
    """

    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_pool = MobileNetV2ConvLayer(config, in_channels=apply_depth_multiplier(config, 320), out_channels=256, kernel_size=1, stride=1, use_normalization=True, use_activation='relu', layer_norm_eps=1e-05)
        self.conv_aspp = MobileNetV2ConvLayer(config, in_channels=apply_depth_multiplier(config, 320), out_channels=256, kernel_size=1, stride=1, use_normalization=True, use_activation='relu', layer_norm_eps=1e-05)
        self.conv_projection = MobileNetV2ConvLayer(config, in_channels=512, out_channels=256, kernel_size=1, stride=1, use_normalization=True, use_activation='relu', layer_norm_eps=1e-05)
        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)
        self.classifier = MobileNetV2ConvLayer(config, in_channels=256, out_channels=config.num_labels, kernel_size=1, use_normalization=False, use_activation=False, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]
        features_pool = self.avg_pool(features)
        features_pool = self.conv_pool(features_pool)
        features_pool = nn.functional.interpolate(features_pool, size=spatial_size, mode='bilinear', align_corners=True)
        features_aspp = self.conv_aspp(features)
        features = torch.cat([features_pool, features_aspp], dim=1)
        features = self.conv_projection(features)
        features = self.dropout(features)
        features = self.classifier(features)
        return features