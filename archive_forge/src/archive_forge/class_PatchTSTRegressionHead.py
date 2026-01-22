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
class PatchTSTRegressionHead(nn.Module):
    """
    Regression head
    """

    def __init__(self, config: PatchTSTConfig, distribution_output=None):
        super().__init__()
        self.y_range = config.output_range
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        self.distribution_output = distribution_output
        head_dim = config.num_input_channels * config.d_model
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        if distribution_output is None:
            self.projection = nn.Linear(head_dim, config.num_targets)
        else:
            self.projection = distribution_output.get_parameter_projection(head_dim)

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                    `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, output_dim)`

        """
        if self.use_cls_token:
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == 'mean':
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == 'max':
            pooled_embedding = embedding.max(dim=2).values
        else:
            raise ValueError(f'pooling operator {self.pooling_type} is not implemented yet')
        pooled_embedding = self.dropout(self.flatten(pooled_embedding))
        output = self.projection(pooled_embedding)
        if (self.distribution_output is None) & (self.y_range is not None):
            output = torch.sigmoid(output) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return output