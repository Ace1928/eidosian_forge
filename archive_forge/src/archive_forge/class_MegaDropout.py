import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaDropout(nn.Module):
    """
    A unified class for standard dropout functionality and featurewise dropout.

    The original fairseq Mega repo used 2 classes for these, which included some unnecessary handling of training logic
    and an unused `inplace` option. The original implementation used torch.nn.functional instead of submodules, which
    is retained here as well.
    """

    def __init__(self, dropout_probability, is_featurewise=False):
        super().__init__()
        self.dropout_probability = dropout_probability
        self.is_featurewise = is_featurewise

    def forward(self, input, batch_first: bool=False):
        if self.is_featurewise:
            if batch_first:
                return F.dropout2d(input.transpose(-1, -2), p=self.dropout_probability, training=self.training).transpose(-1, -2)
            else:
                if input.dim() != 3:
                    raise ValueError('Feature dropout inputs must be exactly 3-dimensional if inputs are ordered [sequence length, batch size, hidden dimension]')
                return F.dropout2d(input.permute(1, 2, 0), p=self.dropout_probability, training=self.training).permute(2, 0, 1)
        else:
            return F.dropout(input, p=self.dropout_probability, training=self.training)