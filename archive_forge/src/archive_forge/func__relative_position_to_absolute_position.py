import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
def _relative_position_to_absolute_position(self, x):
    batch_heads, length, _ = x.size()
    x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])
    x_flat = x.view([batch_heads, length * 2 * length])
    x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])
    x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
    x_final = x_final[:, :length, length - 1:]
    return x_final