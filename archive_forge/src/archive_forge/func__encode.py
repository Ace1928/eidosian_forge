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
def _encode(self, raw_audio, start_level=0, end_level=None):
    if end_level is None:
        end_level = self.levels
    input_audio = raw_audio.permute(0, 2, 1).float()
    latent_states = []
    for level in range(self.levels):
        encoder = self.encoders[level]
        latent_state = encoder(input_audio)
        latent_states.append(latent_state[-1])
    music_tokens = self.bottleneck.encode(latent_states)
    return music_tokens[start_level:end_level]