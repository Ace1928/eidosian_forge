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
class PerceiverMultimodalPreprocessor(AbstractPreprocessor):
    """
    Multimodal preprocessing for Perceiver Encoder.

    Inputs for each modality are preprocessed, then padded with trainable position embeddings to have the same number
    of channels.

    Args:
        modalities (`Mapping[str, PreprocessorType]`):
            Dict mapping modality name to preprocessor.
        mask_probs (`Dict[str, float]`):
            Dict mapping modality name to masking probability of that modality.
        min_padding_size (`int`, *optional*, defaults to 2):
            The minimum padding size for all modalities. The final output will have num_channels equal to the maximum
            channels across all modalities plus min_padding_size.
    """

    def __init__(self, modalities: Mapping[str, PreprocessorType], mask_probs: Optional[Mapping[str, float]]=None, min_padding_size: int=2):
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.min_padding_size = min_padding_size
        self.mask_probs = mask_probs if mask_probs is not None else {}
        self.padding = nn.ParameterDict({modality: nn.Parameter(torch.randn(1, self.num_channels - preprocessor.num_channels)) for modality, preprocessor in modalities.items()})
        self.mask = nn.ParameterDict({modality: nn.Parameter(torch.randn(1, self.num_channels)) for modality, _ in self.mask_probs.items()})

    @property
    def num_channels(self) -> int:
        max_channel_size = max((processor.num_channels for _, processor in self.modalities.items()))
        common_channel_size = max_channel_size + self.min_padding_size
        return common_channel_size

    def forward(self, inputs: Mapping[str, torch.Tensor], pos: Optional[torch.Tensor]=None, network_input_is_1d: bool=True) -> PreprocessorOutputType:
        padded = {}
        modality_sizes = {}
        inputs_without_pos = {}
        for modality, preprocessor in self.modalities.items():
            output, _, inputs_without_pos[modality] = preprocessor(inputs[modality], pos=pos, network_input_is_1d=network_input_is_1d)
            batch_size, num_samples, num_channels = output.shape
            pos_enc = self.padding[modality].expand(batch_size, -1, -1)
            padding = torch.broadcast_to(pos_enc, [batch_size, num_samples, self.num_channels - num_channels])
            output_padded = torch.cat([output, padding], dim=2)
            if modality in self.mask_probs:
                mask_token = self.mask[modality].expand(batch_size, -1, -1)
                mask_prob = self.mask_probs[modality]
                mask = torch.bernoulli(torch.full([batch_size, num_samples], mask_prob))
                mask = torch.unsqueeze(mask, dim=2).to(mask_token.device)
                output_padded = (1 - mask) * output_padded + mask * mask_token
            padded[modality] = output_padded
            modality_sizes[modality] = output_padded.shape[1]
        padded_ls = [padded[k] for k in sorted(padded.keys())]
        final_inputs = torch.cat(padded_ls, dim=1)
        return (final_inputs, modality_sizes, inputs_without_pos)