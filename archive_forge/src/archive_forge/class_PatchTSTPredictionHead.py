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
@add_start_docstrings('The PatchTST for regression Model.', PATCHTST_START_DOCSTRING)
class PatchTSTPredictionHead(nn.Module):

    def __init__(self, config: PatchTSTConfig, num_patches, distribution_output=None):
        super().__init__()
        self.share_projection = config.share_projection
        self.num_input_channels = config.num_input_channels
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        if self.pooling_type or self.use_cls_token:
            head_dim = config.d_model
        else:
            head_dim = config.d_model * num_patches
        if not self.share_projection:
            self.projections = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.num_input_channels):
                self.flattens.append(nn.Flatten(start_dim=2))
                if distribution_output is None:
                    self.projections.append(nn.Linear(head_dim, config.prediction_length))
                else:
                    self.projections.append(distribution_output.get_parameter_projection(head_dim))
                self.dropouts.append(nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity())
        else:
            self.flatten = nn.Flatten(start_dim=2)
            if distribution_output is None:
                self.projection = nn.Linear(head_dim, config.prediction_length)
            else:
                self.projection = distribution_output.get_parameter_projection(head_dim)
            self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, forecast_len, num_channels)`

        """
        if self.use_cls_token:
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == 'mean':
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == 'max':
            pooled_embedding = embedding.max(dim=2).values
        else:
            pooled_embedding = embedding
        if not self.share_projection:
            output = []
            for i in range(self.num_input_channels):
                pooled_embedding = self.flattens[i](pooled_embedding[:, i, :])
                pooled_embedding = self.dropouts[i](pooled_embedding)
                pooled_embedding = self.projections[i](pooled_embedding)
                output.append(pooled_embedding)
            output = torch.stack(output, dim=1)
        else:
            pooled_embedding = self.flatten(pooled_embedding)
            pooled_embedding = self.dropout(pooled_embedding)
            output = self.projection(pooled_embedding)
        if isinstance(output, tuple):
            output = tuple((z.transpose(2, 1) for z in output))
        else:
            output = output.transpose(2, 1)
        return output