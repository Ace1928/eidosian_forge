import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
@staticmethod
def _linear_overlap_add(frames: List[torch.Tensor], stride: int):
    if len(frames) == 0:
        raise ValueError('`frames` cannot be an empty list.')
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]
    frame_length = frames[0].shape[-1]
    time_vec = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
    weight = 0.5 - (time_vec - 0.5).abs()
    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    if sum_weight.min() == 0:
        raise ValueError(f'`sum_weight` minimum element must be bigger than zero: {sum_weight}`')
    return out / sum_weight