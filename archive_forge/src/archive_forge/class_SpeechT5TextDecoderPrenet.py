import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
class SpeechT5TextDecoderPrenet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.positional_dropout)
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_positions = SpeechT5SinusoidalPositionalEmbedding(config.max_text_positions + config.pad_token_id + 1, config.hidden_size, config.pad_token_id)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            raise ValueError('You have to specify `decoder_input_ids`')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        positions = self.embed_positions(input_ids, past_key_values_length)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        inputs_embeds += positions
        inputs_embeds = self.dropout(inputs_embeds)
        return (inputs_embeds, attention_mask)