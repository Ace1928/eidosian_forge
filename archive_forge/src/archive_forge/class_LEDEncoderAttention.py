import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_led import LEDConfig
class LEDEncoderAttention(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.longformer_self_attn = LEDEncoderSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, is_index_masked: Optional[torch.Tensor]=None, is_index_global_attn: Optional[torch.Tensor]=None, is_global_attn: Optional[bool]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        self_outputs = self.longformer_self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
        attn_output = self.output(self_outputs[0])
        outputs = (attn_output,) + self_outputs[1:]
        return outputs