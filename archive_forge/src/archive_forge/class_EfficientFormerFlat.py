import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerFlat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states