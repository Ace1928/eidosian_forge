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
def _init_reassemble_dpt(self, config):
    for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
        self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))
    if config.readout_type == 'project':
        self.readout_projects = nn.ModuleList()
        hidden_size = _get_backbone_hidden_size(config)
        for _ in range(len(config.neck_hidden_sizes)):
            self.readout_projects.append(nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act]))