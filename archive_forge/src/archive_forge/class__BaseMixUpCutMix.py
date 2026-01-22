import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Tuple
import PIL.Image
import torch
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F
from ._transform import _RandomApplyTransform, Transform
from ._utils import _parse_labels_getter, has_any, is_pure_tensor, query_chw, query_size
class _BaseMixUpCutMix(Transform):

    def __init__(self, *, alpha: float=1.0, num_classes: int, labels_getter='default') -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        self.num_classes = num_classes
        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)
        if has_any(flat_inputs, PIL.Image.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask):
            raise ValueError(f'{type(self).__name__}() does not support PIL images, bounding boxes and masks.')
        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f'The labels must be a tensor, but got {type(labels)} instead.')
        elif labels.ndim != 1:
            raise ValueError(f'labels tensor should be of shape (batch_size,) but got shape {labels.shape} instead.')
        params = {'labels': labels, 'batch_size': labels.shape[0], **self._get_params([inpt for inpt, needs_transform in zip(flat_inputs, needs_transform_list) if needs_transform])}
        needs_transform_list[next((idx for idx, inpt in enumerate(flat_inputs) if inpt is labels))] = True
        flat_outputs = [self._transform(inpt, params) if needs_transform else inpt for inpt, needs_transform in zip(flat_inputs, needs_transform_list)]
        return tree_unflatten(flat_outputs, spec)

    def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
        expected_num_dims = 5 if isinstance(inpt, tv_tensors.Video) else 4
        if inpt.ndim != expected_num_dims:
            raise ValueError(f'Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead.')
        if inpt.shape[0] != batch_size:
            raise ValueError(f'The batch size of the image or video does not match the batch size of the labels: {inpt.shape[0]} != {batch_size}.')

    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
        label = one_hot(label, num_classes=self.num_classes)
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))