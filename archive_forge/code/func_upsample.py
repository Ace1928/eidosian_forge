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
@add_start_docstrings('Upsamples a sequence of music tokens using the prior at level `level`.\n\n        Args:\n            music_tokens (`List[torch.LongTensor]` of length `self.levels` ) :\n                A sequence of music tokens which will be used as context to continue the sampling process. Should have\n                `self.levels` tensors, each corresponding to the generation at a certain level.\n        ', JUKEBOX_SAMPLING_INPUT_DOCSTRING)
def upsample(self, music_tokens, labels, **sampling_kwargs) -> List[torch.LongTensor]:
    sample_levels = sampling_kwargs.pop('sample_levels', list(range(len(self.priors) - 1)))
    music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
    return music_tokens