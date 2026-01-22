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
class PerceiverBasicDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention-based decoder. This class can be used to decode the final hidden states of the latents using a
    cross-attention operation, in which the latents produce keys and values.

    The shape of the output of this class depends on how one defines the output queries (also called decoder queries).

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        output_num_channels (`int`, *optional*):
            The number of channels in the output. Will only be used in case *final_project* is set to `True`.
        position_encoding_type (`str`, *optional*, defaults to "trainable"):
            The type of position encoding to use. Can be either "trainable", "fourier", or "none".
        output_index_dims (`int`, *optional*):
            The number of dimensions of the output queries. Ignored if 'position_encoding_type' == 'none'.
        num_channels (`int`, *optional*, defaults to 128):
            The number of channels of the decoder queries. Ignored if 'position_encoding_type' == 'none'.
        qk_channels (`int`, *optional*):
            The number of channels of the queries and keys in the cross-attention layer.
        v_channels (`int`, *optional*):
            The number of channels of the values in the cross-attention layer.
        num_heads (`int`, *optional*, defaults to 1):
            The number of attention heads in the cross-attention layer.
        widening_factor (`int`, *optional*, defaults to 1):
            The widening factor of the cross-attention layer.
        use_query_residual (`bool`, *optional*, defaults to `False`):
            Whether to use a residual connection between the query and the output of the cross-attention layer.
        concat_preprocessed_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the preprocessed input to the query.
        final_project (`bool`, *optional*, defaults to `True`):
            Whether to project the output of the cross-attention layer to a target dimension.
        position_encoding_only (`bool`, *optional*, defaults to `False`):
            Whether to only use this class to define output queries.
    """

    def __init__(self, config: PerceiverConfig, output_num_channels: int, position_encoding_type: Optional[str]='trainable', output_index_dims: Optional[int]=None, num_channels: Optional[int]=128, subsampled_index_dims: Optional[int]=None, qk_channels: Optional[int]=None, v_channels: Optional[int]=None, num_heads: Optional[int]=1, widening_factor: Optional[int]=1, use_query_residual: Optional[bool]=False, concat_preprocessed_input: Optional[bool]=False, final_project: Optional[bool]=True, position_encoding_only: Optional[bool]=False, **position_encoding_kwargs) -> None:
        super().__init__()
        self.output_num_channels = output_num_channels
        self.output_position_encodings = None
        self.position_encoding_type = position_encoding_type
        self.position_encoding_kwargs = position_encoding_kwargs
        if position_encoding_type != 'none':
            self.output_position_encodings, self.positions_projection = build_position_encoding(position_encoding_type=position_encoding_type, **position_encoding_kwargs)
        self.output_index_dims = output_index_dims
        self.num_channels = num_channels
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        self.concat_preprocessed_input = concat_preprocessed_input
        self.final_project = final_project
        self.position_encoding_only = position_encoding_only
        if not self.position_encoding_only:
            self.decoding_cross_attention = PerceiverLayer(config, is_cross_attention=True, qk_channels=qk_channels, v_channels=v_channels, num_heads=num_heads, q_dim=num_channels, kv_dim=config.d_latents, widening_factor=widening_factor, use_query_residual=use_query_residual)
            self.final_layer = nn.Linear(num_channels, output_num_channels) if final_project else nn.Identity()

    @property
    def num_query_channels(self) -> int:
        if self.position_encoding_type == 'none':
            raise ValueError('You cannot calculate number of decoder query channels when position_encoding_type is set to none')
        if self.position_encoding_only:
            if 'project_pos_dim' in self.position_encoding_kwargs:
                return self.position_encoding_kwargs['project_pos_dim']
            return self.output_position_encodings.output_size()
        if self.final_project:
            return self.output_num_channels
        return self.num_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if self.position_encoding_type == 'none':
            raise ValueError('You cannot construct decoder queries when position_encoding_type is set to none')
        if subsampled_points is not None:
            indices = [torch.from_numpy(x) for x in np.unravel_index(subsampled_points.cpu(), self.output_index_dims)]
            pos = torch.stack(indices, dim=1)
            batch_size = inputs.shape[0]
            pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
            pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
            if self.position_encoding_type == 'trainable':
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == 'fourier':
                pos_emb = self.output_position_encodings(self.output_index_dims, batch_size=batch_size, device=inputs.device, dtype=inputs.dtype, pos=pos)
            pos_emb = self.positions_projection(pos_emb)
            pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]
            if self.position_encoding_type == 'trainable':
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == 'fourier':
                pos_emb = self.output_position_encodings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)
            pos_emb = self.positions_projection(pos_emb)
        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError('Value is required for inputs_without_pos if concat_preprocessed_input is True')
            pos_emb = torch.cat([inputs_without_pos, pos_emb], dim=-1)
        return pos_emb

    def forward(self, query: torch.Tensor, z: torch.FloatTensor, query_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> PerceiverDecoderOutput:
        cross_attentions = () if output_attentions else None
        layer_outputs = self.decoding_cross_attention(query, attention_mask=query_mask, head_mask=None, inputs=z, inputs_mask=None, output_attentions=output_attentions)
        output = layer_outputs[0]
        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs[1],)
        logits = self.final_layer(output)
        return PerceiverDecoderOutput(logits=logits, cross_attentions=cross_attentions)