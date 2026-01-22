import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
from ....utils import logging
from .configuration_mctct import MCTCTConfig
def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
    if len(attention_mask.shape) > 2:
        attention_mask = attention_mask[:, :, -1]
    subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
    bsz = attention_mask.size()[0]
    attention_mask = torch.zeros((bsz, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
    attention_mask[torch.arange(bsz, device=attention_mask.device), subsampled_lengths - 1] = 1
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).long()
    return attention_mask