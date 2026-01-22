import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
@dataclass
class RealmReaderOutput(ModelOutput):
    """
    Outputs of [`RealmReader`] models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Total loss.
        retriever_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Retriever loss.
        reader_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Reader loss.
        retriever_correct (`torch.BoolTensor` of shape `(config.searcher_beam_size,)`, *optional*):
            Whether or not an evidence block contains answer.
        reader_correct (`torch.BoolTensor` of shape `(config.reader_beam_size, num_candidates)`, *optional*):
            Whether or not a span candidate contains answer.
        block_idx (`torch.LongTensor` of shape `()`):
            The index of the retrieved evidence block in which the predicted answer is most likely.
        candidate (`torch.LongTensor` of shape `()`):
            The index of the retrieved span candidates in which the predicted answer is most likely.
        start_pos (`torch.IntTensor` of shape `()`):
            Predicted answer starting position in *RealmReader*'s inputs.
        end_pos (`torch.IntTensor` of shape `()`):
            Predicted answer ending position in *RealmReader*'s inputs.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: torch.FloatTensor = None
    retriever_loss: torch.FloatTensor = None
    reader_loss: torch.FloatTensor = None
    retriever_correct: torch.BoolTensor = None
    reader_correct: torch.BoolTensor = None
    block_idx: torch.LongTensor = None
    candidate: torch.LongTensor = None
    start_pos: torch.int32 = None
    end_pos: torch.int32 = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None