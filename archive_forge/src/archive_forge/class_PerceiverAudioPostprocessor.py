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
class PerceiverAudioPostprocessor(nn.Module):
    """
    Audio postprocessing for Perceiver. Can be used to convert the decoder output to audio features.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        in_channels (`int`):
            Number of channels in the input.
        postproc_type (`str`, *optional*, defaults to `"patches"`):
            Postprocessor type to use. Currently, only "patches" is supported.
    """

    def __init__(self, config: PerceiverConfig, in_channels: int, postproc_type: str='patches') -> None:
        super().__init__()
        if postproc_type not in ('patches',):
            raise ValueError('Invalid postproc_type!')
        self.classifier = nn.Linear(in_channels, config.samples_per_patch)

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor]=None, modality_sizes=None) -> torch.Tensor:
        logits = self.classifier(inputs)
        return torch.reshape(logits, [inputs.shape[0], -1])