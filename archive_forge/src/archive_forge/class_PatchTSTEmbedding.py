import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
class PatchTSTEmbedding(nn.Module):

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_input_channels = config.num_input_channels
        self.share_embedding = config.share_embedding
        if self.share_embedding:
            self.input_embedding = nn.Linear(config.patch_length, config.d_model)
        else:
            self.input_embedding = nn.ModuleList()
            for _ in range(config.num_input_channels):
                self.input_embedding.append(nn.Linear(config.patch_length, config.d_model))

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        num_input_channels = patch_input.shape[1]
        if num_input_channels != self.num_input_channels:
            raise ValueError(f'The defined number of input channels ({self.num_input_channels}) in the config has to be the same as the number of channels in the batch input ({num_input_channels})')
        if self.share_embedding:
            embeddings = self.input_embedding(patch_input)
        else:
            embeddings = [self.input_embedding[i](patch_input[:, i, :, :]) for i in range(num_input_channels)]
            embeddings = torch.stack(embeddings, dim=1)
        return embeddings