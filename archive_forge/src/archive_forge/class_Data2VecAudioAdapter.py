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
class Data2VecAudioAdapter(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None
        self.layers = nn.ModuleList((Data2VecAudioAdapterLayer(config) for _ in range(config.num_adapter_layers)))
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or layerdrop_prob > self.layerdrop:
                hidden_states = layer(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states