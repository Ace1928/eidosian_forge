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
class JukeboxBottleneck(nn.Module):

    def __init__(self, config, levels):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(JukeboxBottleneckBlock(config))

    def encode(self, raw_audio):
        music_tokens = [level_block.encode(hidden_states) for level_block, hidden_states in zip(self.level_blocks, raw_audio)]
        return music_tokens

    def decode(self, music_tokens, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        quantised_audio = [level_block.decode(z) for level_block, z in zip(self.level_blocks[start_level:end_level], music_tokens)]
        return quantised_audio

    def forward(self, input_audio):
        music_tokens, quantised_states, commit_losses, metrics = ([], [], [], [])
        for level in range(self.levels):
            level_block = self.level_blocks[-level - 1]
            hidden_states = input_audio[level]
            sampled_tokens, quantised_state, commit_loss, metric = level_block(hidden_states, update_codebook=self.training)
            music_tokens.append(sampled_tokens)
            if not self.training:
                quantised_state = quantised_state.detach()
            quantised_states.append(quantised_state)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return (music_tokens, quantised_states, commit_losses, metrics)