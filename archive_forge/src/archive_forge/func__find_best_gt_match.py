import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import IntTensor, Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
@staticmethod
def _find_best_gt_match(thr: int, gt_matches: Tensor, idx_iou: float, gt_ignore: Tensor, ious: Tensor, idx_det: int) -> int:
    """Return id of best ground truth match with current detection.

        Args:
            thr:
                Current threshold value.
            gt_matches:
                Tensor showing if a ground truth matches for threshold ``t`` exists.
            idx_iou:
                Id of threshold ``t``.
            gt_ignore:
                Tensor showing if ground truth should be ignored.
            ious:
                IoUs for all combinations of detection and ground truth.
            idx_det:
                Id of current detection.

        """
    previously_matched = gt_matches[idx_iou]
    remove_mask = previously_matched | gt_ignore
    gt_ious = ious[idx_det] * ~remove_mask
    match_idx = gt_ious.argmax().item()
    if gt_ious[match_idx] > thr:
        return match_idx
    return -1