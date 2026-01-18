import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
@staticmethod
def torch_gather_b2(params, indices):
    if params.shape[:2] != indices.shape[:2]:
        raise ValueError(f'Make sure that the first two dimensions of params and indices are identical,                 but they are params: {params.shape[:2]} vs. indices: {indices.shape[:2]}')
    num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
    num_indices_to_pick_from = params.shape[2]
    shift = torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
    indices_shift = torch.div(shift, num_indices_to_gather, rounding_mode='floor') * num_indices_to_pick_from
    flattened_indices = indices.view(-1) + indices_shift
    flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])
    out_flattened = flattened_params.index_select(0, flattened_indices)
    out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
    return out