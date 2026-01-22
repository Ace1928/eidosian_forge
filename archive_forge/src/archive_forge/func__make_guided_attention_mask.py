import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
@staticmethod
def _make_guided_attention_mask(input_length, output_length, sigma, device):
    grid_y, grid_x = torch.meshgrid(torch.arange(input_length, device=device), torch.arange(output_length, device=device), indexing='xy')
    grid_x = grid_x.float() / output_length
    grid_y = grid_y.float() / input_length
    return 1.0 - torch.exp(-(grid_y - grid_x) ** 2 / (2 * sigma ** 2))