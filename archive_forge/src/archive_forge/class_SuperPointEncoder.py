from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
class SuperPointEncoder(nn.Module):
    """
    SuperPoint encoder module. It is made of 4 convolutional layers with ReLU activation and max pooling, reducing the
     dimensionality of the image.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        self.input_dim = 1
        conv_blocks = []
        conv_blocks.append(SuperPointConvBlock(config, self.input_dim, config.encoder_hidden_sizes[0], add_pooling=True))
        for i in range(1, len(config.encoder_hidden_sizes) - 1):
            conv_blocks.append(SuperPointConvBlock(config, config.encoder_hidden_sizes[i - 1], config.encoder_hidden_sizes[i], add_pooling=True))
        conv_blocks.append(SuperPointConvBlock(config, config.encoder_hidden_sizes[-2], config.encoder_hidden_sizes[-1], add_pooling=False))
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, input, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None
        for conv_block in self.conv_blocks:
            input = conv_block(input)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (input,)
        output = input
        if not return_dict:
            return tuple((v for v in [output, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=output, hidden_states=all_hidden_states)