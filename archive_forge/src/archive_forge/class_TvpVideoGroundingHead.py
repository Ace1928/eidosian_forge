import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
class TvpVideoGroundingHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer_0 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.layer_1 = nn.Linear(config.hidden_size * 2, 2)
        self.activation_0 = nn.ReLU()
        self.activation_1 = nn.Sigmoid()

    def forward(self, pooler_output):
        logits = self.activation_0(self.layer_0(pooler_output))
        logits = self.activation_1(self.layer_1(logits))
        return logits