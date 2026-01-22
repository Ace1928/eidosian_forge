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
class PerceiverProjectionPostprocessor(nn.Module):
    """
    Projection postprocessing for Perceiver. Can be used to project the channels of the decoder output to a lower
    dimension.

    Args:
        in_channels (`int`):
            Number of channels in the input.
        out_channels (`int`):
            Number of channels in the output.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor]=None, modality_sizes=None) -> torch.Tensor:
        logits = self.classifier(inputs)
        return logits