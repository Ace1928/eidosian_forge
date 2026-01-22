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
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
def _subsample_labels(self, label):
    """
        Randomly sample a subset of positive and negative examples, and overwrite the label vector to the ignore value
        (-1) for all elements that are not included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
    pos_idx, neg_idx = subsample_labels(label, self.batch_size_per_image, self.positive_fraction, 0)
    label.fill_(-1)
    label.scatter_(0, pos_idx, 1)
    label.scatter_(0, neg_idx, 0)
    return label