import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
@dataclass
class PatchTSTModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Parameters:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`, *optional*)
            Bool masked tensor indicating which patches are masked
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Mean of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Std of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input to the Transformer
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask: torch.FloatTensor = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None
    patch_input: torch.FloatTensor = None