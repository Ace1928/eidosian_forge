import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
def _tie_weights(self):
    if self.config.tie_word_embeddings:
        self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
        self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
        self._tie_or_clone_weights(self.lm_head, self.shared)