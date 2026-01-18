from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
@WRAPPER_FACTORIES.register(datasets.WIDERFace)
def widerface_wrapper(dataset, target_keys):
    target_keys = parse_target_keys(target_keys, available={'bbox', 'blur', 'expression', 'illumination', 'occlusion', 'pose', 'invalid'}, default='all')

    def wrapper(idx, sample):
        image, target = sample
        if target is None:
            return (image, target)
        target = {key: target[key] for key in target_keys}
        if 'bbox' in target_keys:
            target['bbox'] = F.convert_bounding_box_format(tv_tensors.BoundingBoxes(target['bbox'], format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(image.height, image.width)), new_format=tv_tensors.BoundingBoxFormat.XYXY)
        return (image, target)
    return wrapper