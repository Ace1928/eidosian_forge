import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
@staticmethod
def _init_pe(config: PatchTSMixerConfig) -> nn.Parameter:
    if config.positional_encoding_type == 'random':
        position_enc = nn.Parameter(torch.randn(config.num_patches, config.d_model), requires_grad=True)
    elif config.positional_encoding_type == 'sincos':
        position_enc = torch.zeros(config.num_patches, config.d_model)
        position = torch.arange(0, config.num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)
        position_enc = position_enc - position_enc.mean()
        position_enc = position_enc / (position_enc.std() * 10)
        position_enc = nn.Parameter(position_enc, requires_grad=False)
    else:
        raise ValueError(f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'.")
    return position_enc