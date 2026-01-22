import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig
class AlignVisionBlock(nn.Module):
    """
    This corresponds to the block module of original the EfficientNet vision encoder implementation.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int, expand_ratio: int, kernel_size: int, drop_rate: float, id_skip: bool, adjust_padding: bool):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.expand = True if self.expand_ratio != 1 else False
        expand_in_dim = in_dim * expand_ratio
        if self.expand:
            self.expansion = AlignVisionExpansionLayer(config=config, in_dim=in_dim, out_dim=expand_in_dim, stride=stride)
        self.depthwise_conv = AlignVisionDepthwiseLayer(config=config, in_dim=expand_in_dim if self.expand else in_dim, stride=stride, kernel_size=kernel_size, adjust_padding=adjust_padding)
        self.squeeze_excite = AlignVisionSqueezeExciteLayer(config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand)
        self.projection = AlignVisionFinalBlockLayer(config=config, in_dim=expand_in_dim if self.expand else in_dim, out_dim=out_dim, stride=stride, drop_rate=drop_rate, id_skip=id_skip)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        embeddings = hidden_states
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.squeeze_excite(hidden_states)
        hidden_states = self.projection(embeddings, hidden_states)
        return hidden_states