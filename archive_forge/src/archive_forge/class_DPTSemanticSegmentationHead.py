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
class DPTSemanticSegmentationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        features = config.fusion_hidden_size
        self.head = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(features), nn.ReLU(), nn.Dropout(config.semantic_classifier_dropout), nn.Conv2d(features, config.num_labels, kernel_size=1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states[self.config.head_in_index]
        logits = self.head(hidden_states)
        return logits