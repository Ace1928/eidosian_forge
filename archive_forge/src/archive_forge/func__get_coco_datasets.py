import contextlib
import io
import json
from types import ModuleType
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch import distributed as dist
from typing_extensions import Literal
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator, _validate_iou_type_arg
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _get_coco_datasets(self, average: Literal['macro', 'micro']) -> Tuple[object, object]:
    """Returns the coco datasets for the target and the predictions."""
    if average == 'micro':
        groundtruth_labels = apply_to_collection(self.groundtruth_labels, Tensor, lambda x: torch.zeros_like(x))
        detection_labels = apply_to_collection(self.detection_labels, Tensor, lambda x: torch.zeros_like(x))
    else:
        groundtruth_labels = self.groundtruth_labels
        detection_labels = self.detection_labels
    coco_target, coco_preds = (self.coco(), self.coco())
    coco_target.dataset = self._get_coco_format(labels=groundtruth_labels, boxes=self.groundtruth_box if len(self.groundtruth_box) > 0 else None, masks=self.groundtruth_mask if len(self.groundtruth_mask) > 0 else None, crowds=self.groundtruth_crowds, area=self.groundtruth_area)
    coco_preds.dataset = self._get_coco_format(labels=detection_labels, boxes=self.detection_box if len(self.detection_box) > 0 else None, masks=self.detection_mask if len(self.detection_mask) > 0 else None, scores=self.detection_scores)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_target.createIndex()
        coco_preds.createIndex()
    return (coco_preds, coco_target)