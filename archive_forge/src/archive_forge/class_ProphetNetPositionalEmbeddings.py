import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_prophetnet import ProphetNetConfig
class ProphetNetPositionalEmbeddings(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """

    def __init__(self, config: ProphetNetConfig) -> None:
        self.max_length = config.max_position_embeddings
        super().__init__(config.max_position_embeddings, config.hidden_size, config.pad_token_id)

    def forward(self, inputs_shape, device, attention_mask=None, past_key_values=None, position_ids=None):
        assert position_ids is None or self.padding_idx is None, 'If position_ids is pre-computed then padding_idx should not be set.'
        if position_ids is None:
            if past_key_values is not None:
                prev_num_input_ids = past_key_values[0][0].shape[2]
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                position_ids = torch.ones((1, 1), dtype=torch.long, device=device) * int(self.padding_idx + num_input_ids)
            else:
                if attention_mask is None:
                    attention_mask = torch.ones(inputs_shape, dtype=torch.long, device=device)
                position_ids = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() + self.padding_idx
                position_ids = position_ids.clamp(0, self.max_length - 1)
        return (super().forward(position_ids), position_ids)

    def _forward(self, position_ids):
        return super().forward(position_ids)