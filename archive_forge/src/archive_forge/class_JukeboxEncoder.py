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
class JukeboxEncoder(nn.Module):

    def __init__(self, config, width, depth, levels, downs_t, strides_t):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for i, down_t, stride_t in iterator:
            self.level_blocks.append(JukeboxEncoderConvBlock(config, config.conv_input_shape if i == 0 else config.embed_dim, width, depth, down_t, stride_t))

    def forward(self, hidden_states):
        all_hidden_states = []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            hidden_states = level_block(hidden_states)
            all_hidden_states.append(hidden_states)
        return all_hidden_states