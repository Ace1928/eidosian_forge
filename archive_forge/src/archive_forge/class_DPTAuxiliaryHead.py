import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
class DPTAuxiliaryHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        features = config.fusion_hidden_size
        self.head = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(features), nn.ReLU(), nn.Dropout(0.1, False), nn.Conv2d(features, config.num_labels, kernel_size=1))

    def forward(self, hidden_states):
        logits = self.head(hidden_states)
        return logits