import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_deformable_detr import DeformableDetrConfig
def to(self, device):
    cast_tensor = self.tensors.to(device)
    mask = self.mask
    if mask is not None:
        cast_mask = mask.to(device)
    else:
        cast_mask = None
    return NestedTensor(cast_tensor, cast_mask)