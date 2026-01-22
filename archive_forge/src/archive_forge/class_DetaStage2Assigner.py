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
class DetaStage2Assigner(nn.Module):

    def __init__(self, num_queries, max_k=4):
        super().__init__()
        self.positive_fraction = 0.25
        self.bg_label = 400
        self.batch_size_per_image = num_queries
        self.proposal_matcher = DetaMatcher(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True)
        self.k = max_k

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

    def forward(self, outputs, targets, return_cost_matrix=False):
        bs = len(targets)
        indices = []
        ious = []
        for b in range(bs):
            iou, _ = box_iou(center_to_corners_format(targets[b]['boxes']), center_to_corners_format(outputs['init_reference'][b].detach()))
            matched_idxs, matched_labels = self.proposal_matcher(iou)
            sampled_idxs, sampled_gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets[b]['class_labels'])
            pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            indices.append((pos_pr_inds, pos_gt_inds))
            ious.append(iou)
        if return_cost_matrix:
            return (indices, ious)
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)