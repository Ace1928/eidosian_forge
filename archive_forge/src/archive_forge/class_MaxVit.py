import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
class MaxVit(nn.Module):
    """
    Implements MaxVit Transformer from the `MaxViT: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`_ paper.
    Args:
        input_size (Tuple[int, int]): Size of the input image.
        stem_channels (int): Number of channels in the stem.
        partition_size (int): Size of the partitions.
        block_channels (List[int]): Number of channels in each block.
        block_layers (List[int]): Number of layers in each block.
        stochastic_depth_prob (float): Probability of stochastic depth. Expands to a list of probabilities for each layer that scales linearly to the specified value.
        squeeze_ratio (float): Squeeze ratio in the SE Layer. Default: 0.25.
        expansion_ratio (float): Expansion ratio in the MBConv bottleneck. Default: 4.
        norm_layer (Callable[..., nn.Module]): Normalization function. Default: None (setting to None will produce a `BatchNorm2d(eps=1e-3, momentum=0.99)`).
        activation_layer (Callable[..., nn.Module]): Activation function Default: nn.GELU.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Expansion ratio of the MLP layer. Default: 4.
        mlp_dropout (float): Dropout probability for the MLP layer. Default: 0.0.
        attention_dropout (float): Dropout probability for the attention layer. Default: 0.0.
        num_classes (int): Number of classes. Default: 1000.
    """

    def __init__(self, input_size: Tuple[int, int], stem_channels: int, partition_size: int, block_channels: List[int], block_layers: List[int], head_dim: int, stochastic_depth_prob: float, norm_layer: Optional[Callable[..., nn.Module]]=None, activation_layer: Callable[..., nn.Module]=nn.GELU, squeeze_ratio: float=0.25, expansion_ratio: float=4, mlp_ratio: int=4, mlp_dropout: float=0.0, attention_dropout: float=0.0, num_classes: int=1000) -> None:
        super().__init__()
        _log_api_usage_once(self)
        input_channels = 3
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.99)
        block_input_sizes = _make_block_input_shapes(input_size, len(block_channels))
        for idx, block_input_size in enumerate(block_input_sizes):
            if block_input_size[0] % partition_size != 0 or block_input_size[1] % partition_size != 0:
                raise ValueError(f'Input size {block_input_size} of block {idx} is not divisible by partition size {partition_size}. Consider changing the partition size or the input size.\nCurrent configuration yields the following block input sizes: {block_input_sizes}.')
        self.stem = nn.Sequential(Conv2dNormActivation(input_channels, stem_channels, 3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer, bias=False, inplace=None), Conv2dNormActivation(stem_channels, stem_channels, 3, stride=1, norm_layer=None, activation_layer=None, bias=True))
        input_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        self.partition_size = partition_size
        self.blocks = nn.ModuleList()
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels
        p_stochastic = np.linspace(0, stochastic_depth_prob, sum(block_layers)).tolist()
        p_idx = 0
        for in_channel, out_channel, num_layers in zip(in_channels, out_channels, block_layers):
            self.blocks.append(MaxVitBlock(in_channels=in_channel, out_channels=out_channel, squeeze_ratio=squeeze_ratio, expansion_ratio=expansion_ratio, norm_layer=norm_layer, activation_layer=activation_layer, head_dim=head_dim, mlp_ratio=mlp_ratio, mlp_dropout=mlp_dropout, attention_dropout=attention_dropout, partition_size=partition_size, input_grid_size=input_size, n_layers=num_layers, p_stochastic=p_stochastic[p_idx:p_idx + num_layers]))
            input_size = self.blocks[-1].grid_size
            p_idx += num_layers
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(block_channels[-1]), nn.Linear(block_channels[-1], block_channels[-1]), nn.Tanh(), nn.Linear(block_channels[-1], num_classes, bias=False))
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)