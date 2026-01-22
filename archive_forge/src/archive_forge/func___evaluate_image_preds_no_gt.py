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
def __evaluate_image_preds_no_gt(self, det: Tensor, idx: int, det_label_mask: Tensor, max_det: int, area_range: Tuple[int, int], num_iou_thrs: int) -> Dict[str, Any]:
    """Evaluate images with a prediction but no ground truth."""
    num_gt = 0
    gt_ignore = torch.zeros(num_gt, dtype=torch.bool, device=self.device)
    det = [det[i] for i in det_label_mask]
    scores = self.detection_scores[idx]
    scores_filtered = scores[det_label_mask]
    scores_sorted, dtind = torch.sort(scores_filtered, descending=True)
    det = [det[i] for i in dtind]
    if len(det) > max_det:
        det = det[:max_det]
    num_det = len(det)
    det_areas = compute_area(det, iou_type=self.iou_type).to(self.device)
    det_ignore_area = (det_areas < area_range[0]) | (det_areas > area_range[1])
    ar = det_ignore_area.reshape((1, num_det))
    det_ignore = torch.repeat_interleave(ar, num_iou_thrs, 0)
    return {'dtMatches': torch.zeros((num_iou_thrs, num_det), dtype=torch.bool, device=self.device), 'gtMatches': torch.zeros((num_iou_thrs, num_gt), dtype=torch.bool, device=self.device), 'dtScores': scores_sorted.to(self.device), 'gtIgnore': gt_ignore.to(self.device), 'dtIgnore': det_ignore.to(self.device)}