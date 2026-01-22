import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
class EsmFoldStructureModuleTransitionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.linear_1 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init='relu')
        self.linear_2 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init='relu')
        self.linear_3 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init='final')
        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        return s