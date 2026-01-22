import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
class PatchTSTPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: PatchTSTConfig, num_patches: int):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.num_input_channels = config.num_input_channels
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
            num_patches += 1
        self.position_enc = self._init_pe(config, num_patches)
        self.positional_dropout = nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()

    @staticmethod
    def _init_pe(config: PatchTSTConfig, num_patches: int) -> nn.Parameter:
        if config.positional_encoding_type == 'random':
            position_enc = nn.Parameter(torch.randn(num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == 'sincos':
            position_enc = torch.zeros(num_patches, config.d_model)
            position = torch.arange(0, num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'.")
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        if self.use_cls_token:
            patch_input = self.positional_dropout(patch_input + self.position_enc[1:, :])
            cls_token = self.cls_token + self.position_enc[:1, :]
            cls_tokens = cls_token.expand(patch_input.shape[0], self.num_input_channels, -1, -1)
            hidden_state = torch.cat((cls_tokens, patch_input), dim=2)
        else:
            hidden_state = self.positional_dropout(patch_input + self.position_enc)
        return hidden_state