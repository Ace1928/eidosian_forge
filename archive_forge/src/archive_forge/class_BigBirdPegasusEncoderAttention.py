import copy
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_bigbird_pegasus import BigBirdPegasusConfig
class BigBirdPegasusEncoderAttention(nn.Module):

    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config
        self.seed = seed
        self.attention_type = config.attention_type
        if self.attention_type == 'original_full':
            self.self = BigBirdPegasusSelfAttention(config)
        elif self.attention_type == 'block_sparse':
            self.self = BigBirdPegasusBlockSparseAttention(config, seed)
        else:
            raise ValueError(f'attention_type can either be original_full or block_sparse, but is {self.config.attention_type}')
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)

    def set_attention_type(self, value: str):
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        if value == 'original_full':
            attn_weights = BigBirdPegasusSelfAttention(self.config)
        else:
            attn_weights = BigBirdPegasusBlockSparseAttention(self.config, self.seed)
        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value
        if not self.training:
            self.self.eval()

    def forward(self, hidden_states, attention_mask=None, head_mask=None, past_key_value=None, output_attentions=False, band_mask=None, from_mask=None, to_mask=None, from_blocked_mask=None, to_blocked_mask=None):
        head_mask = head_mask.reshape(1, -1, 1, 1) if head_mask is not None else None
        if self.config.attention_type == 'original_full':
            self_outputs = self.self(hidden_states, attention_mask, head_mask, past_key_value=past_key_value, output_attentions=output_attentions)
        else:
            self_outputs = self.self(hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions)
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]
        return outputs