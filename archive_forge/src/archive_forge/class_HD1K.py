import itertools
import os
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from ..io.image import _read_png_16
from .utils import _read_pfm, verify_str_arg
from .vision import VisionDataset
class HD1K(FlowDataset):
    """`HD1K <http://hci-benchmark.iwr.uni-heidelberg.de/>`__ dataset for optical flow.

    The dataset is expected to have the following structure: ::

        root
            hd1k
                hd1k_challenge
                    image_2
                hd1k_flow_gt
                    flow_occ
                hd1k_input
                    image_2

    Args:
        root (string): Root directory of the HD1K Dataset.
        split (string, optional): The dataset split, either "train" (default) or "test"
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
    """
    _has_builtin_flow_mask = True

    def __init__(self, root: str, split: str='train', transforms: Optional[Callable]=None) -> None:
        super().__init__(root=root, transforms=transforms)
        verify_str_arg(split, 'split', valid_values=('train', 'test'))
        root = Path(root) / 'hd1k'
        if split == 'train':
            for seq_idx in range(36):
                flows = sorted(glob(str(root / 'hd1k_flow_gt' / 'flow_occ' / f'{seq_idx:06d}_*.png')))
                images = sorted(glob(str(root / 'hd1k_input' / 'image_2' / f'{seq_idx:06d}_*.png')))
                for i in range(len(flows) - 1):
                    self._flow_list += [flows[i]]
                    self._image_list += [[images[i], images[i + 1]]]
        else:
            images1 = sorted(glob(str(root / 'hd1k_challenge' / 'image_2' / '*10.png')))
            images2 = sorted(glob(str(root / 'hd1k_challenge' / 'image_2' / '*11.png')))
            for image1, image2 in zip(images1, images2):
                self._image_list += [[image1, image2]]
        if not self._image_list:
            raise FileNotFoundError('Could not find the HD1K images. Please make sure the directory structure is correct.')

    def _read_flow(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        return _read_16bits_png_with_flow_and_valid_mask(file_name)

    def __getitem__(self, index: int) -> Union[T1, T2]:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 4-tuple with ``(img1, img2, flow, valid_flow_mask)`` where ``valid_flow_mask``
            is a numpy boolean mask of shape (H, W)
            indicating which flow values are valid. The flow is a numpy array of
            shape (2, H, W) and the images are PIL images. ``flow`` and ``valid_flow_mask`` are None if
            ``split="test"``.
        """
        return super().__getitem__(index)