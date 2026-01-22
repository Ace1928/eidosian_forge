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