import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_yoso import YosoConfig
def to_contiguous(input_tensors):
    if isinstance(input_tensors, list):
        out = []
        for tensor in input_tensors:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            out.append(tensor)
        return out
    else:
        if not input_tensors.is_contiguous():
            input_tensors = input_tensors.contiguous()
        return input_tensors