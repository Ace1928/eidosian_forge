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
class EsmFoldResidueMLP(nn.Module):

    def __init__(self, embed_dim, inner_dim, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, inner_dim), nn.ReLU(), nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.mlp(x)