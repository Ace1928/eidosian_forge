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
class JukeboxMusicTokenConditioner(nn.Module):
    """
    The `JukeboxMusicTokenConditioner` takes music tokens as an input (coresponding to the codes of the VQVAE's
    codebook) and upsamples it using a single layer of decoder convolution block (the same is used in the VQVAE).
    """

    def __init__(self, config, level):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.music_vocab_size, config.hidden_size)
        config.embed_dim = config.music_vocab_size
        self.upsampler = JukeboxDecoderConvBock(config, config.hidden_size, config.res_conv_width, config.res_conv_depth, config.res_downs_t[level], config.res_strides_t[level], reverse_dilation=False)
        self.layer_norm = JukeboxLayerNorm(config.hidden_size)

    def forward(self, music_tokens, raw_audio_conditionning=None):
        """
        Args:
            music_tokens (`torch.LongTensor`):
                Music tokens form the uper level in range(nb_discrete_codes)
            raw_audio_conditionning (`torch.LongTensor`, *optional*):
                Audio used when primed sampling, raw audio information that conditions the generation
        """
        if raw_audio_conditionning is None:
            raw_audio_conditionning = 0.0
        music_tokens = music_tokens.long()
        hidden_states = self.embed_tokens(music_tokens)
        hidden_states = hidden_states + raw_audio_conditionning
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.upsampler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states