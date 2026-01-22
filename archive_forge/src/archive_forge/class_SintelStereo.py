import functools
import json
import os
import random
import shutil
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Callable, cast, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from .utils import _read_pfm, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
class SintelStereo(StereoMatchingDataset):
    """Sintel `Stereo Dataset <http://sintel.is.tue.mpg.de/stereo>`_.

    The dataset is expected to have the following structure: ::

        root
            Sintel
                training
                    final_left
                        scene1
                            img1.png
                            img2.png
                            ...
                        ...
                    final_right
                        scene2
                            img1.png
                            img2.png
                            ...
                        ...
                    disparities
                        scene1
                            img1.png
                            img2.png
                            ...
                        ...
                    occlusions
                        scene1
                            img1.png
                            img2.png
                            ...
                        ...
                    outofframe
                        scene1
                            img1.png
                            img2.png
                            ...
                        ...

    Args:
        root (string): Root directory where Sintel Stereo is located.
        pass_name (string): The name of the pass to use, either "final", "clean" or "both".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """
    _has_built_in_disparity_mask = True

    def __init__(self, root: str, pass_name: str='final', transforms: Optional[Callable]=None) -> None:
        super().__init__(root, transforms)
        verify_str_arg(pass_name, 'pass_name', valid_values=('final', 'clean', 'both'))
        root = Path(root) / 'Sintel'
        pass_names = {'final': ['final'], 'clean': ['clean'], 'both': ['final', 'clean']}[pass_name]
        for p in pass_names:
            left_img_pattern = str(root / 'training' / f'{p}_left' / '*' / '*.png')
            right_img_pattern = str(root / 'training' / f'{p}_right' / '*' / '*.png')
            self._images += self._scan_pairs(left_img_pattern, right_img_pattern)
            disparity_pattern = str(root / 'training' / 'disparities' / '*' / '*.png')
            self._disparities += self._scan_pairs(disparity_pattern, None)

    def _get_occlussion_mask_paths(self, file_path: str) -> Tuple[str, str]:
        fpath = Path(file_path)
        basename = fpath.name
        scenedir = fpath.parent
        sampledir = scenedir.parent.parent
        occlusion_path = str(sampledir / 'occlusions' / scenedir.name / basename)
        outofframe_path = str(sampledir / 'outofframe' / scenedir.name / basename)
        if not os.path.exists(occlusion_path):
            raise FileNotFoundError(f'Occlusion mask {occlusion_path} does not exist')
        if not os.path.exists(outofframe_path):
            raise FileNotFoundError(f'Out of frame mask {outofframe_path} does not exist')
        return (occlusion_path, outofframe_path)

    def _read_disparity(self, file_path: str) -> Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]]:
        if file_path is None:
            return (None, None)
        disparity_map = np.asarray(Image.open(file_path), dtype=np.float32)
        r, g, b = np.split(disparity_map, 3, axis=-1)
        disparity_map = r * 4 + g / 2 ** 6 + b / 2 ** 14
        disparity_map = np.transpose(disparity_map, (2, 0, 1))
        occlued_mask_path, out_of_frame_mask_path = self._get_occlussion_mask_paths(file_path)
        valid_mask = np.asarray(Image.open(occlued_mask_path)) == 0
        off_mask = np.asarray(Image.open(out_of_frame_mask_path)) == 0
        valid_mask = np.logical_and(off_mask, valid_mask)
        return (disparity_map, valid_mask)

    def __getitem__(self, index: int) -> T2:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 4-tuple with ``(img_left, img_right, disparity, valid_mask)`` is returned.
            The disparity is a numpy array of shape (1, H, W) and the images are PIL images whilst
            the valid_mask is a numpy array of shape (H, W).
        """
        return cast(T2, super().__getitem__(index))