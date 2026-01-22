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
class Data2VecAudioPositionalConvEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Data2VecAudioPositionalConvLayer(config) for _ in range(config.num_conv_pos_embeddings)])

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states