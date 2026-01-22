from typing import Optional
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_regnet import RegNetConfig
class RegNetEmbeddings(nn.Module):
    """
    RegNet Embedddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: RegNetConfig):
        super().__init__()
        self.embedder = RegNetConvLayer(config.num_channels, config.embedding_size, kernel_size=3, stride=2, activation=config.hidden_act)
        self.num_channels = config.num_channels

    def forward(self, pixel_values):
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        hidden_state = self.embedder(pixel_values)
        return hidden_state