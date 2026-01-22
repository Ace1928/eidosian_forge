import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
class LevitConvEmbeddings(nn.Module):
    """
    LeViT Conv Embeddings with Batch Norm, used in the initial patch embedding layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, embeddings):
        embeddings = self.convolution(embeddings)
        embeddings = self.batch_norm(embeddings)
        return embeddings