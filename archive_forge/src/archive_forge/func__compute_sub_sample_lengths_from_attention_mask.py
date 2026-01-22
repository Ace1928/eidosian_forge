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
def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
    kernel_size, stride = (self.config.adaptor_kernel_size, self.config.adaptor_stride)
    pad = kernel_size // 2
    seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
    seq_lens = (seq_lens + 2 * pad - kernel_size) / stride + 1
    return seq_lens.floor()