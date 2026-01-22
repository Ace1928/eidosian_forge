import copy
import math
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_bart import BartConfig
def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)