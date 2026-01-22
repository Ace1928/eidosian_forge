import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
    max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
    batch_size = len(tensors)
    batch_shape = [batch_size] + max_size
    b, _, h, w = batch_shape
    dtype = tensors[0].dtype
    device = tensors[0].device
    padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
    padding_masks = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
        padded_tensor[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]].copy_(tensor)
        padding_mask[:tensor.shape[1], :tensor.shape[2]] = False
    return (padded_tensors, padding_masks)