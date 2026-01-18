from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
@WRAPPER_FACTORIES.register(datasets.VOCDetection)
def voc_detection_wrapper_factory(dataset, target_keys):
    target_keys = parse_target_keys(target_keys, available={'annotation', 'boxes', 'labels'}, default={'boxes', 'labels'})

    def wrapper(idx, sample):
        image, target = sample
        batched_instances = list_of_dicts_to_dict_of_lists(target['annotation']['object'])
        if 'annotation' not in target_keys:
            target = {}
        if 'boxes' in target_keys:
            target['boxes'] = tv_tensors.BoundingBoxes([[int(bndbox[part]) for part in ('xmin', 'ymin', 'xmax', 'ymax')] for bndbox in batched_instances['bndbox']], format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(image.height, image.width))
        if 'labels' in target_keys:
            target['labels'] = torch.tensor([VOC_DETECTION_CATEGORY_TO_IDX[category] for category in batched_instances['name']])
        return (image, target)
    return wrapper