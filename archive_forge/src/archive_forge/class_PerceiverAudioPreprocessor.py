import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
class PerceiverAudioPreprocessor(AbstractPreprocessor):
    """
    Audio preprocessing for Perceiver Encoder.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        prep_type (`str`, *optional*, defaults to `"patches"`):
            Preprocessor type to use. Only "patches" is supported.
        samples_per_patch (`int`, *optional*, defaults to 96):
            Number of samples per patch.
        position_encoding_type (`str`, *optional*, defaults to `"fourier"`):
            Type of position encoding to use. Can be "trainable" or "fourier".
        concat_or_add_pos (`str`, *optional*, defaults to `"concat"`):
            How to concatenate the position encoding to the input. Can be "concat" or "add".
        out_channels (`int`, *optional*, defaults to 64):
            Number of channels in the output.
        project_pos_dim (`int`, *optional*, defaults to -1):
            Dimension of the position encoding to project to. If -1, no projection is applied.
        **position_encoding_kwargs (`Dict`, *optional*):
            Keyword arguments for the position encoding.
    """

    def __init__(self, config, prep_type: str='patches', samples_per_patch: int=96, position_encoding_type: str='fourier', concat_or_add_pos: str='concat', out_channels=64, project_pos_dim=-1, **position_encoding_kwargs):
        super().__init__()
        self.config = config
        if prep_type not in ('patches',):
            raise ValueError(f"Prep_type {prep_type} is invalid, can only be 'patches'.")
        if concat_or_add_pos not in ['concat', 'add']:
            raise ValueError(f"Concat_or_pos {concat_or_add_pos} is invalid, can only be 'concat' or 'add'.")
        self.samples_per_patch = samples_per_patch
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(position_encoding_type=position_encoding_type, out_channels=out_channels, project_pos_dim=project_pos_dim, **position_encoding_kwargs)

    @property
    def num_channels(self) -> int:
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == 'add':
            return pos_dim
        return self.samples_per_patch + pos_dim

    def _build_network_inputs(self, inputs):
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        if self.position_encoding_type == 'trainable':
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == 'fourier':
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)
        pos_enc = self.positions_projection(pos_enc)
        if self.concat_or_add_pos == 'concat':
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == 'add':
            inputs_with_pos = inputs + pos_enc
        return (inputs_with_pos, inputs)

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor]=None, network_input_is_1d: bool=True):
        inputs = torch.reshape(inputs, [inputs.shape[0], -1, self.samples_per_patch])
        inputs, inputs_without_pos = self._build_network_inputs(inputs)
        modality_sizes = None
        return (inputs, modality_sizes, inputs_without_pos)