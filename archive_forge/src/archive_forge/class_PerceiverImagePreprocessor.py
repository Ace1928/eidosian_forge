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
class PerceiverImagePreprocessor(AbstractPreprocessor):
    """
    Image preprocessing for Perceiver Encoder.

    Note: the *out_channels* argument refers to the output channels of a convolutional layer, if *prep_type* is set to
    "conv1x1" or "conv". If one adds absolute position embeddings, one must make sure the *num_channels* of the
    position encoding kwargs are set equal to the *out_channels*.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        prep_type (`str`, *optional*, defaults to `"conv"`):
            Preprocessing type. Can be "conv1x1", "conv", "patches", "pixels".
        spatial_downsample (`int`, *optional*, defaults to 4):
            Spatial downsampling factor.
        temporal_downsample (`int`, *optional*, defaults to 1):
            Temporal downsampling factor (only relevant in case a time dimension is present).
        position_encoding_type (`str`, *optional*, defaults to `"fourier"`):
            Position encoding type. Can be "fourier" or "trainable".
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input.
        out_channels (`int`, *optional*, defaults to 64):
            Number of channels in the output.
        conv_after_patching (`bool`, *optional*, defaults to `False`):
            Whether to apply a convolutional layer after patching.
        conv_after_patching_in_channels (`int`, *optional*, defaults to 54):
            Number of channels in the input of the convolutional layer after patching.
        conv2d_use_batchnorm (`bool`, *optional*, defaults to `True`):
            Whether to use batch normalization in the convolutional layer.
        concat_or_add_pos (`str`, *optional*, defaults to `"concat"`):
            How to concatenate the position encoding to the input. Can be "concat" or "add".
        project_pos_dim (`int`, *optional*, defaults to -1):
            Dimension of the position encoding to project to. If -1, no projection is applied.
        **position_encoding_kwargs (`Dict`, *optional*):
            Keyword arguments for the position encoding.
    """

    def __init__(self, config, prep_type='conv', spatial_downsample: int=4, temporal_downsample: int=1, position_encoding_type: str='fourier', in_channels: int=3, out_channels: int=64, conv_after_patching: bool=False, conv_after_patching_in_channels: int=54, conv2d_use_batchnorm: bool=True, concat_or_add_pos: str='concat', project_pos_dim: int=-1, **position_encoding_kwargs):
        super().__init__()
        self.config = config
        if prep_type not in ('conv', 'patches', 'pixels', 'conv1x1'):
            raise ValueError(f'Prep_type {prep_type} is invalid')
        if concat_or_add_pos not in ['concat', 'add']:
            raise ValueError(f'Invalid value {concat_or_add_pos} for concat_or_add_pos.')
        self.in_channels = in_channels
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels
        if self.prep_type == 'conv':
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError('Only powers of 4 expected for spatial and 1 expected for temporal downsampling with conv.')
            self.convnet = Conv2DDownsample(in_channels=in_channels, num_layers=int(convnet_num_layers), out_channels=out_channels, use_batchnorm=conv2d_use_batchnorm)
        elif self.prep_type == 'conv1x1':
            if temporal_downsample != 1:
                raise ValueError('Conv1x1 does not downsample in time.')
            self.convnet_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(spatial_downsample, spatial_downsample))
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(position_encoding_type=position_encoding_type, out_channels=out_channels, project_pos_dim=project_pos_dim, **position_encoding_kwargs)
        self.conv_after_patches = nn.Linear(conv_after_patching_in_channels, self.out_channels) if conv_after_patching else nn.Identity()

    @property
    def num_channels(self) -> int:
        is_temporal = self.position_embeddings.num_dimensions > 2
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == 'add':
            return pos_dim
        if self.conv_after_patching or self.prep_type in ('conv1x1', 'conv'):
            inp_dim = self.out_channels
        elif self.prep_type == 'pixels':
            inp_dim = self.in_channels
            if not is_temporal:
                inp_dim = math.ceil(inp_dim / self.spatial_downsample)
        elif self.prep_type == 'patches':
            if self.conv_after_patching:
                inp_dim = self.out_channels
            else:
                inp_dim = self.in_channels * self.spatial_downsample ** 2
                if is_temporal:
                    inp_dim *= self.temporal_downsample
        return inp_dim + pos_dim

    def _build_network_inputs(self, inputs: torch.Tensor, network_input_is_1d: bool=True):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.

        """
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])
        if self.position_encoding_type == 'trainable':
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == 'fourier':
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)
        pos_enc = self.positions_projection(pos_enc)
        if not network_input_is_1d:
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])
        if self.concat_or_add_pos == 'concat':
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == 'add':
            inputs_with_pos = inputs + pos_enc
        return (inputs_with_pos, inputs)

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor]=None, network_input_is_1d: bool=True):
        if self.prep_type == 'conv':
            inputs = self.convnet(inputs)
        elif self.prep_type == 'conv1x1':
            inputs = self.convnet_1x1(inputs)
        elif self.prep_type == 'pixels':
            if inputs.ndim == 4:
                inputs = inputs[::self.spatial_downsample, ::self.spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[:, ::self.temporal_downsample, :, ::self.spatial_downsample, ::self.spatial_downsample]
            else:
                raise ValueError('Unsupported data format for pixels.')
        elif self.prep_type == 'patches':
            inputs = space_to_depth(inputs, temporal_block_size=self.temporal_downsample, spatial_block_size=self.spatial_downsample)
            if inputs.ndim == 5 and inputs.shape[1] == 1:
                inputs = inputs.squeeze(dim=1)
            inputs = self.conv_after_patches(inputs)
        if self.prep_type != 'patches':
            if inputs.ndim == 4:
                inputs = inputs.permute(0, 2, 3, 1)
            elif inputs.ndim == 5:
                inputs = inputs.permute(0, 1, 3, 4, 2)
            else:
                raise ValueError('Unsupported data format for conv1x1.')
        inputs, inputs_without_pos = self._build_network_inputs(inputs, network_input_is_1d)
        modality_sizes = None
        return (inputs, modality_sizes, inputs_without_pos)