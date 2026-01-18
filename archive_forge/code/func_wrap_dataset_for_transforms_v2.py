from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
def wrap_dataset_for_transforms_v2(dataset, target_keys=None):
    """[BETA] Wrap a ``torchvision.dataset`` for usage with :mod:`torchvision.transforms.v2`.

    .. v2betastatus:: wrap_dataset_for_transforms_v2 function

    Example:
        >>> dataset = torchvision.datasets.CocoDetection(...)
        >>> dataset = wrap_dataset_for_transforms_v2(dataset)

    .. note::

       For now, only the most popular datasets are supported. Furthermore, the wrapper only supports dataset
       configurations that are fully supported by ``torchvision.transforms.v2``. If you encounter an error prompting you
       to raise an issue to ``torchvision`` for a dataset or configuration that you need, please do so.

    The dataset samples are wrapped according to the description below.

    Special cases:

        * :class:`~torchvision.datasets.CocoDetection`: Instead of returning the target as list of dicts, the wrapper
          returns a dict of lists. In addition, the key-value-pairs ``"boxes"`` (in ``XYXY`` coordinate format),
          ``"masks"`` and ``"labels"`` are added and wrap the data in the corresponding ``torchvision.tv_tensors``.
          The original keys are preserved. If ``target_keys`` is omitted, returns only the values for the
          ``"image_id"``, ``"boxes"``, and ``"labels"``.
        * :class:`~torchvision.datasets.VOCDetection`: The key-value-pairs ``"boxes"`` and ``"labels"`` are added to
          the target and wrap the data in the corresponding ``torchvision.tv_tensors``. The original keys are
          preserved. If ``target_keys`` is omitted, returns only the values for the ``"boxes"`` and ``"labels"``.
        * :class:`~torchvision.datasets.CelebA`: The target for ``target_type="bbox"`` is converted to the ``XYXY``
          coordinate format and wrapped into a :class:`~torchvision.tv_tensors.BoundingBoxes` tv_tensor.
        * :class:`~torchvision.datasets.Kitti`: Instead returning the target as list of dicts, the wrapper returns a
          dict of lists. In addition, the key-value-pairs ``"boxes"`` and ``"labels"`` are added and wrap the data
          in the corresponding ``torchvision.tv_tensors``. The original keys are preserved. If ``target_keys`` is
          omitted, returns only the values for the ``"boxes"`` and ``"labels"``.
        * :class:`~torchvision.datasets.OxfordIIITPet`: The target for ``target_type="segmentation"`` is wrapped into a
          :class:`~torchvision.tv_tensors.Mask` tv_tensor.
        * :class:`~torchvision.datasets.Cityscapes`: The target for ``target_type="semantic"`` is wrapped into a
          :class:`~torchvision.tv_tensors.Mask` tv_tensor. The target for ``target_type="instance"`` is *replaced* by
          a dictionary with the key-value-pairs ``"masks"`` (as :class:`~torchvision.tv_tensors.Mask` tv_tensor) and
          ``"labels"``.
        * :class:`~torchvision.datasets.WIDERFace`: The value for key ``"bbox"`` in the target is converted to ``XYXY``
          coordinate format and wrapped into a :class:`~torchvision.tv_tensors.BoundingBoxes` tv_tensor.

    Image classification datasets

        This wrapper is a no-op for image classification datasets, since they were already fully supported by
        :mod:`torchvision.transforms` and thus no change is needed for :mod:`torchvision.transforms.v2`.

    Segmentation datasets

        Segmentation datasets, e.g. :class:`~torchvision.datasets.VOCSegmentation`, return a two-tuple of
        :class:`PIL.Image.Image`'s. This wrapper leaves the image as is (first item), while wrapping the
        segmentation mask into a :class:`~torchvision.tv_tensors.Mask` (second item).

    Video classification datasets

        Video classification datasets, e.g. :class:`~torchvision.datasets.Kinetics`, return a three-tuple containing a
        :class:`torch.Tensor` for the video and audio and a :class:`int` as label. This wrapper wraps the video into a
        :class:`~torchvision.tv_tensors.Video` while leaving the other items as is.

        .. note::

            Only datasets constructed with ``output_format="TCHW"`` are supported, since the alternative
            ``output_format="THWC"`` is not supported by :mod:`torchvision.transforms.v2`.

    Args:
        dataset: the dataset instance to wrap for compatibility with transforms v2.
        target_keys: Target keys to return in case the target is a dictionary. If ``None`` (default), selected keys are
            specific to the dataset. If ``"all"``, returns the full target. Can also be a collection of strings for
            fine grained access. Currently only supported for :class:`~torchvision.datasets.CocoDetection`,
            :class:`~torchvision.datasets.VOCDetection`, :class:`~torchvision.datasets.Kitti`, and
            :class:`~torchvision.datasets.WIDERFace`. See above for details.
    """
    if not (target_keys is None or target_keys == 'all' or (isinstance(target_keys, collections.abc.Collection) and all((isinstance(key, str) for key in target_keys)))):
        raise ValueError(f"`target_keys` can be None, 'all', or a collection of strings denoting the keys to be returned, but got {target_keys}")
    wrapped_dataset_cls = type(f'Wrapped{type(dataset).__name__}', (VisionDatasetTVTensorWrapper, type(dataset)), {})
    return wrapped_dataset_cls(dataset, target_keys)