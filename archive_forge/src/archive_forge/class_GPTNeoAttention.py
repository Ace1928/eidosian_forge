import os
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from .configuration_gpt_neo import GPTNeoConfig
class GPTNeoAttention(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]
        if self.attention_type in ['global', 'local']:
            self.attention = GPT_NEO_ATTENTION_CLASSES[config._attn_implementation](config, self.attention_type)
        else:
            raise NotImplementedError(f"Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: {config.attention_layers}. Select attn layer types from ['global', 'local'] only.")

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        return self.attention(hidden_states, attention_mask=attention_mask, layer_past=layer_past, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)