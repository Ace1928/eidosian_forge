import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
class DetRandomPadAug(DetAugmenter):
    """Random padding augmenter.

    Parameters
    ----------
    aspect_ratio_range : tuple of floats, default=(0.75, 1.33)
        The padded area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats, default=(1.0, 3.0)
        The padded area of the image must be larger than the original area
    max_attempts : int, default=50
        Number of attempts at generating a padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.
    pad_val: float or tuple of float, default=(128, 128, 128)
        pixel value to be filled when padding is enabled.
    """

    def __init__(self, aspect_ratio_range=(0.75, 1.33), area_range=(1.0, 3.0), max_attempts=50, pad_val=(128, 128, 128)):
        if not isinstance(pad_val, (list, tuple)):
            assert isinstance(pad_val, numeric_types)
            pad_val = pad_val
        if not isinstance(aspect_ratio_range, (list, tuple)):
            assert isinstance(aspect_ratio_range, numeric_types)
            logging.info('Using fixed aspect ratio: %s in DetRandomPadAug', str(aspect_ratio_range))
            aspect_ratio_range = (aspect_ratio_range, aspect_ratio_range)
        if not isinstance(area_range, (tuple, list)):
            assert isinstance(area_range, numeric_types)
            logging.info('Using fixed area range: %s in DetRandomPadAug', area_range)
            area_range = (area_range, area_range)
        super(DetRandomPadAug, self).__init__(aspect_ratio_range=aspect_ratio_range, area_range=area_range, max_attempts=max_attempts, pad_val=pad_val)
        self.pad_val = pad_val
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self.enabled = False
        if area_range[1] <= 1.0 or area_range[0] > area_range[1]:
            warnings.warn('Skip DetRandomPadAug due to invalid parameters: %s', area_range)
        elif aspect_ratio_range[0] <= 0 or aspect_ratio_range[0] > aspect_ratio_range[1]:
            warnings.warn('Skip DetRandomPadAug due to invalid aspect_ratio_range: %s', aspect_ratio_range)
        else:
            self.enabled = True

    def __call__(self, src, label):
        """Augmenter body"""
        height, width, _ = src.shape
        pad = self._random_pad_proposal(label, height, width)
        if pad:
            x, y, w, h, label = pad
            src = copyMakeBorder(src, y, h - y - height, x, w - x - width, 16, values=self.pad_val)
        return (src, label)

    def _update_labels(self, label, pad_box, height, width):
        """Update label according to padding region"""
        out = label.copy()
        out[:, (1, 3)] = (out[:, (1, 3)] * width + pad_box[0]) / pad_box[2]
        out[:, (2, 4)] = (out[:, (2, 4)] * height + pad_box[1]) / pad_box[3]
        return out

    def _random_pad_proposal(self, label, height, width):
        """Generate random padding region"""
        from math import sqrt
        if not self.enabled or height <= 0 or width <= 0:
            return ()
        min_area = self.area_range[0] * height * width
        max_area = self.area_range[1] * height * width
        for _ in range(self.max_attempts):
            ratio = random.uniform(*self.aspect_ratio_range)
            if ratio <= 0:
                continue
            h = int(round(sqrt(min_area / ratio)))
            max_h = int(round(sqrt(max_area / ratio)))
            if round(h * ratio) < width:
                h = int((width + 0.499999) / ratio)
            if h < height:
                h = height
            if h > max_h:
                h = max_h
            if h < max_h:
                h = random.randint(h, max_h)
            w = int(round(h * ratio))
            if h - height < 2 or w - width < 2:
                continue
            y = random.randint(0, max(0, h - height))
            x = random.randint(0, max(0, w - width))
            new_label = self._update_labels(label, (x, y, w, h), height, width)
            return (x, y, w, h, new_label)
        return ()