import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
class FlavaITMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pooler = FlavaPooler(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = self.pooler(x)
        x = self.seq_relationship(x)
        return x