import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_data2vec_audio import Data2VecAudioConfig
class Data2VecAudioPositionalConvLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.conv_pos_kernel_size, padding=config.conv_pos_kernel_size // 2, groups=config.num_conv_pos_embedding_groups)
        self.padding = Data2VecAudioPadLayer(config.conv_pos_kernel_size)
        self.activation = ACT2FN[config.feat_extract_activation]
        self.layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        return hidden_states