import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mgp_str import MgpstrConfig
@dataclass
class MgpstrModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        logits (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`):
            Tuple of `torch.FloatTensor` (one for the output of character of shape `(batch_size,
            config.max_token_length, config.num_character_labels)`, + one for the output of bpe of shape `(batch_size,
            config.max_token_length, config.num_bpe_labels)`, + one for the output of wordpiece of shape `(batch_size,
            config.max_token_length, config.num_wordpiece_labels)`) .

            Classification scores (before SoftMax) of character, bpe and wordpiece.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, config.max_token_length,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        a3_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`):
            Tuple of `torch.FloatTensor` (one for the attention of character, + one for the attention of bpe`, + one
            for the attention of wordpiece) of shape `(batch_size, config.max_token_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    logits: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    a3_attentions: Optional[Tuple[torch.FloatTensor]] = None