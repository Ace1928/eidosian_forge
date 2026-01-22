import math
import warnings
from collections.abc import Sequence
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sew_d import SEWDConfig
class SEWDUpsampling(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.hidden_size, config.hidden_size * config.squeeze_factor)
        self.activation = ACT2FN[config.feat_extract_activation]
        self.squeeze_factor = config.squeeze_factor

    def forward(self, hidden_states):
        hidden_states = self.projection(hidden_states)
        hidden_states = self.activation(hidden_states)
        if self.squeeze_factor > 1:
            bsz, src_len, src_embed_dim = hidden_states.size()
            tgt_len = src_len * self.squeeze_factor
            tgt_embed_dim = src_embed_dim // self.squeeze_factor
            hidden_states = hidden_states.reshape(bsz, src_len, self.squeeze_factor, tgt_embed_dim)
            hidden_states = hidden_states.reshape(bsz, tgt_len, tgt_embed_dim)
        return hidden_states