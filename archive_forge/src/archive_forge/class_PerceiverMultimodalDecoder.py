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
class PerceiverMultimodalDecoder(PerceiverAbstractDecoder):
    """
    Multimodal decoding by composing uni-modal decoders. The *modalities* argument of the constructor is a dictionary
    mapping modality name to the decoder of that modality. That decoder will be used to construct queries for that
    modality. Modality-specific queries are padded with trainable modality-specific parameters, after which they are
    concatenated along the time dimension.

    Next, there is a shared cross attention operation across all modalities.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        modalities (`Dict[str, PerceiverAbstractDecoder]`):
            Dictionary mapping modality name to the decoder of that modality.
        num_outputs (`int`):
            The number of outputs of the decoder.
        output_num_channels (`int`):
            The number of channels in the output.
        min_padding_size (`int`, *optional*, defaults to 2):
            The minimum padding size for all modalities. The final output will have num_channels equal to the maximum
            channels across all modalities plus min_padding_size.
        subsampled_index_dims (`Dict[str, PerceiverAbstractDecoder]`, *optional*):
            Dictionary mapping modality name to the subsampled index dimensions to use for the decoder query of that
            modality.
    """

    def __init__(self, config: PerceiverConfig, modalities: Dict[str, PerceiverAbstractDecoder], num_outputs: int, output_num_channels: int, min_padding_size: Optional[int]=2, subsampled_index_dims: Optional[Dict[str, PerceiverAbstractDecoder]]=None, **decoder_kwargs) -> None:
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.subsampled_index_dims = subsampled_index_dims
        self.min_padding_size = min_padding_size
        self.output_num_channels = output_num_channels
        self.num_outputs = num_outputs
        self.decoder = PerceiverBasicDecoder(config, output_index_dims=(num_outputs,), output_num_channels=output_num_channels, position_encoding_type='none', num_channels=self.num_query_channels, **decoder_kwargs)
        self.padding = nn.ParameterDict({modality: nn.Parameter(torch.randn(1, self.num_query_channels - decoder.num_query_channels)) for modality, decoder in modalities.items()})

    @property
    def num_query_channels(self) -> int:
        max_channel_size = max((decoder.num_query_channels for _, decoder in self.modalities.items()))
        common_channel_size = max_channel_size + self.min_padding_size
        return common_channel_size

    def decoder_query(self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None):
        inputs = restructure(modality_sizes, inputs)
        subsampled_points = subsampled_points or {}
        decoder_queries = {}
        for modality, decoder in self.modalities.items():
            input_without_pos = None
            if inputs_without_pos is not None:
                input_without_pos = inputs_without_pos.get(modality, None)
            query = decoder.decoder_query(inputs=inputs[modality], modality_sizes=None, inputs_without_pos=input_without_pos, subsampled_points=subsampled_points.get(modality, None))
            decoder_queries[modality] = query

        def embed(modality, x):
            x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = self.padding[modality]
            pos = torch.broadcast_to(pos, [x.shape[0], x.shape[1], self.num_query_channels - x.shape[2]])
            return torch.cat([x, pos], dim=2)
        return torch.cat([embed(modality, decoder_queries[modality]) for modality in sorted(self.modalities.keys())], dim=1)

    def forward(self, query: torch.Tensor, z: torch.FloatTensor, query_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> torch.Tensor:
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        return decoder_outputs