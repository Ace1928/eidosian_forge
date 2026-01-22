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
def _get_coco_format(self, labels: List[torch.Tensor], boxes: Optional[List[torch.Tensor]]=None, masks: Optional[List[torch.Tensor]]=None, scores: Optional[List[torch.Tensor]]=None, crowds: Optional[List[torch.Tensor]]=None, area: Optional[List[torch.Tensor]]=None) -> Dict:
    """Transforms and returns all cached targets or predictions in COCO format.

        Format is defined at
        https://cocodataset.org/#format-data

        """
    images = []
    annotations = []
    annotation_id = 1
    for image_id, image_labels in enumerate(labels):
        if boxes is not None:
            image_boxes = boxes[image_id]
            image_boxes = image_boxes.cpu().tolist()
        if masks is not None:
            image_masks = masks[image_id]
            if len(image_masks) == 0 and boxes is None:
                continue
        image_labels = image_labels.cpu().tolist()
        images.append({'id': image_id})
        if 'segm' in self.iou_type and len(image_masks) > 0:
            images[-1]['height'], images[-1]['width'] = (image_masks[0][0][0], image_masks[0][0][1])
        for k, image_label in enumerate(image_labels):
            if boxes is not None:
                image_box = image_boxes[k]
            if masks is not None and len(image_masks) > 0:
                image_mask = image_masks[k]
                image_mask = {'size': image_mask[0], 'counts': image_mask[1]}
            if 'bbox' in self.iou_type and len(image_box) != 4:
                raise ValueError(f'Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})')
            if not isinstance(image_label, int):
                raise ValueError(f'Invalid input class of sample {image_id}, element {k} (expected value of type integer, got type {type(image_label)})')
            area_stat_box = None
            area_stat_mask = None
            if area is not None and area[image_id][k].cpu().tolist() > 0:
                area_stat = area[image_id][k].cpu().tolist()
            else:
                area_stat = self.mask_utils.area(image_mask) if 'segm' in self.iou_type else image_box[2] * image_box[3]
                if len(self.iou_type) > 1:
                    area_stat_box = image_box[2] * image_box[3]
                    area_stat_mask = self.mask_utils.area(image_mask)
            annotation = {'id': annotation_id, 'image_id': image_id, 'area': area_stat, 'category_id': image_label, 'iscrowd': crowds[image_id][k].cpu().tolist() if crowds is not None else 0}
            if area_stat_box is not None:
                annotation['area_bbox'] = area_stat_box
                annotation['area_segm'] = area_stat_mask
            if boxes is not None:
                annotation['bbox'] = image_box
            if masks is not None:
                annotation['segmentation'] = image_mask
            if scores is not None:
                score = scores[image_id][k].cpu().tolist()
                if not isinstance(score, float):
                    raise ValueError(f'Invalid input score of sample {image_id}, element {k} (expected value of type float, got type {type(score)})')
                annotation['score'] = score
            annotations.append(annotation)
            annotation_id += 1
    classes = [{'id': i, 'name': str(i)} for i in self._get_classes()]
    return {'images': images, 'annotations': annotations, 'categories': classes}