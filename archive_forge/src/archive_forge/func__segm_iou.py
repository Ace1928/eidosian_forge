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
def _segm_iou(det: List[Tuple[np.ndarray, np.ndarray]], gt: List[Tuple[np.ndarray, np.ndarray]]) -> Tensor:
    """Compute IOU between detections and ground-truths using mask-IOU.

    Implementation is based on pycocotools toolkit for mask_utils.

    Args:
       det: A list of detection masks as ``[(RLE_SIZE, RLE_COUNTS)]``, where ``RLE_SIZE`` is (width, height) dimension
           of the input and RLE_COUNTS is its RLE representation;

       gt: A list of ground-truth masks as ``[(RLE_SIZE, RLE_COUNTS)]``, where ``RLE_SIZE`` is (width, height) dimension
           of the input and RLE_COUNTS is its RLE representation;

    """
    import pycocotools.mask as mask_utils
    det_coco_format = [{'size': i[0], 'counts': i[1]} for i in det]
    gt_coco_format = [{'size': i[0], 'counts': i[1]} for i in gt]
    return torch.tensor(mask_utils.iou(det_coco_format, gt_coco_format, [False for _ in gt]))