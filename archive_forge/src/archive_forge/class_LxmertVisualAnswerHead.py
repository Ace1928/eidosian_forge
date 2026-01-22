import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
class LxmertVisualAnswerHead(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2), GeLU(), nn.LayerNorm(hid_dim * 2, eps=1e-12), nn.Linear(hid_dim * 2, num_labels))

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)