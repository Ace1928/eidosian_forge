import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
class BigBirdAttention(nn.Module):

    def __init__(self, config, seed=None):
        super().__init__()
        self.attention_type = config.attention_type
        self.config = config
        self.seed = seed
        if self.config.attention_type == 'original_full':
            self.self = BigBirdSelfAttention(config)
        elif self.config.attention_type == 'block_sparse':
            self.self = BigBirdBlockSparseAttention(config, seed)
        else:
            raise ValueError(f'attention_type can either be original_full or block_sparse, but is {self.config.attention_type}')
        self.output = BigBirdSelfOutput(config)

    def set_attention_type(self, value: str):
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        if value == 'original_full':
            attn_weights = BigBirdSelfAttention(self.config)
        else:
            attn_weights = BigBirdBlockSparseAttention(self.config, self.seed)
        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value
        if not self.training:
            self.self.eval()

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, band_mask=None, from_mask=None, to_mask=None, from_blocked_mask=None, to_blocked_mask=None):
        if band_mask is not None:
            band_mask = band_mask.to(hidden_states.dtype)
        if from_mask is not None:
            from_mask = from_mask.to(hidden_states.dtype)
        if to_mask is not None:
            to_mask = to_mask.to(hidden_states.dtype)
        if self.attention_type == 'original_full':
            self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        else:
            if encoder_hidden_states is not None:
                raise ValueError("BigBird cannot be used as a decoder when config.attention_type != 'original_full'")
            self_outputs = self.self(hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs