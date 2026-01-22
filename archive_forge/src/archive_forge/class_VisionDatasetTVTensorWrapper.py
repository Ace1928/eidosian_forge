from __future__ import annotations
import collections.abc
import contextlib
from collections import defaultdict
from copy import copy
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
class VisionDatasetTVTensorWrapper:

    def __init__(self, dataset, target_keys):
        dataset_cls = type(dataset)
        if not isinstance(dataset, datasets.VisionDataset):
            raise TypeError(f"This wrapper is meant for subclasses of `torchvision.datasets.VisionDataset`, but got a '{dataset_cls.__name__}' instead.\nFor an example of how to perform the wrapping for custom datasets, see\n\nhttps://pytorch.org/vision/main/auto_examples/plot_tv_tensors.html#do-i-have-to-wrap-the-output-of-the-datasets-myself")
        for cls in dataset_cls.mro():
            if cls in WRAPPER_FACTORIES:
                wrapper_factory = WRAPPER_FACTORIES[cls]
                if target_keys is not None and cls not in {datasets.CocoDetection, datasets.VOCDetection, datasets.Kitti, datasets.WIDERFace}:
                    raise ValueError(f'`target_keys` is currently only supported for `CocoDetection`, `VOCDetection`, `Kitti`, and `WIDERFace`, but got {cls.__name__}.')
                break
            elif cls is datasets.VisionDataset:
                msg = f'No wrapper exists for dataset class {dataset_cls.__name__}. Please wrap the output yourself.'
                if dataset_cls in datasets.__dict__.values():
                    msg = f'{msg} If an automated wrapper for this dataset would be useful for you, please open an issue at https://github.com/pytorch/vision/issues.'
                raise TypeError(msg)
        self._dataset = dataset
        self._target_keys = target_keys
        self._wrapper = wrapper_factory(dataset, target_keys)
        self.transform, dataset.transform = (dataset.transform, None)
        self.target_transform, dataset.target_transform = (dataset.target_transform, None)
        self.transforms, dataset.transforms = (dataset.transforms, None)

    def __getattr__(self, item):
        with contextlib.suppress(AttributeError):
            return object.__getattribute__(self, item)
        return getattr(self._dataset, item)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        sample = self._wrapper(idx, sample)
        if self.transforms is not None:
            sample = self.transforms(*sample)
        return sample

    def __len__(self):
        return len(self._dataset)

    def __reduce__(self):
        dataset = copy(self._dataset)
        dataset.transform = self.transform
        dataset.transforms = self.transforms
        dataset.target_transform = self.target_transform
        return (wrap_dataset_for_transforms_v2, (dataset, self._target_keys))