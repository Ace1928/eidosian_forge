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
def _suff_cache_len(self):
    """
        Precondition:
            key and value are appended with the current context and self.sample_t reflects the 1-indexed sample
            location in the context.
        """
    previous_block_length = (self.sample_t - 1) % self.block_ctx + 1 + self.block_ctx
    REQUIRED_CACHE_LEN = {'dense_attn': self.sample_t, 'block_attn': (self.sample_t - 1) % self.block_ctx + 1, 'transpose_block_attn': self.sample_t, 'prev_block_attn': self.sample_t if self.sample_t <= self.block_ctx else previous_block_length, 'cross_attn': self.encoder_len, 'prime_attn': min(self.sample_t, self._encoder_len)}
    return REQUIRED_CACHE_LEN[self.attn_func]