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
class JukeboxDecoderConvBock(nn.Module):

    def __init__(self, config, embed_dim, hidden_dim, depth, down_t, stride_t, reverse_dilation=True):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t = stride_t * 2
            pad_t = stride_t // 2
            self.proj_in = nn.Conv1d(embed_dim, hidden_dim, 3, 1, 1)
            for i in range(down_t):
                blocks.append(JukeboxResnet1D(config, hidden_dim, depth, reverse_dilation))
                blocks.append(nn.ConvTranspose1d(hidden_dim, hidden_dim if i < down_t - 1 else embed_dim, filter_t, stride_t, pad_t))
        self.upsample_block = nn.ModuleList(blocks)

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        for block in self.upsample_block:
            hidden_states = block(hidden_states)
        return hidden_states