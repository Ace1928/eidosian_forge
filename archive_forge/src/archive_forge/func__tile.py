import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def _tile(self, hidden_states):
    dim, embed_width = hidden_states.shape
    if dim < self.nb_discrete_codes:
        n_repeats = (self.nb_discrete_codes + dim - 1) // dim
        std = 0.01 / np.sqrt(embed_width)
        hidden_states = hidden_states.repeat(n_repeats, 1)
        hidden_states = hidden_states + torch.randn_like(hidden_states) * std
    return hidden_states