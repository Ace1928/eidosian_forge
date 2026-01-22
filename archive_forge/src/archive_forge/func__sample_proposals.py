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
def _sample_proposals(self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor):
    """
        Based on the matching between N proposals and M groundtruth, sample the proposals and set their classification
        labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N). Tensor: a vector of the same length,
            the classification label for
                each sampled proposal. Each sample is labeled as either a category in [0, num_classes) or the
                background (num_classes).
        """
    has_gt = gt_classes.numel() > 0
    if has_gt:
        gt_classes = gt_classes[matched_idxs]
        gt_classes[matched_labels == 0] = self.bg_label
        gt_classes[matched_labels == -1] = -1
    else:
        gt_classes = torch.zeros_like(matched_idxs) + self.bg_label
    sampled_fg_idxs, sampled_bg_idxs = subsample_labels(gt_classes, self.batch_size_per_image, self.positive_fraction, self.bg_label)
    sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
    return (sampled_idxs, gt_classes[sampled_idxs])