import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gpt_bigcode import GPTBigCodeConfig
def _get_mask_value(self, device, dtype):
    if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
        self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
    return self.mask_value