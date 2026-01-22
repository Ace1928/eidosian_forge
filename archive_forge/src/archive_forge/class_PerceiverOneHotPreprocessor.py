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
class PerceiverOneHotPreprocessor(AbstractPreprocessor):
    """
    One-hot preprocessor for Perceiver Encoder. Can be used to add a dummy index dimension to the input.

    Args:
        config ([`PerceiverConfig`]):
            Model configuration.
    """

    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        self.config: PerceiverConfig = config

    @property
    def num_channels(self) -> int:
        return self.config.num_labels

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor]=None, network_input_is_1d: bool=True):
        inputs = inputs[:, None, :]
        return (inputs, None, inputs)