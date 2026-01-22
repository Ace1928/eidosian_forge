import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
class RandomSizedCropAug(Augmenter):
    """Make random crop with random resizing and random aspect ratio jitter augmenter.

    Parameters
    ----------
    size : tuple of (int, int)
        Size of the crop formatted as (width, height).
    area : float in (0, 1] or tuple of (float, float)
        If tuple, minimum area and maximum area to be maintained after cropping
        If float, minimum area to be maintained after cropping, maximum area is set to 1.0
    ratio : tuple of (float, float)
        Aspect ratio range as (min_aspect_ratio, max_aspect_ratio)
    interp: int, optional, default=2
        Interpolation method. See resize_short for details.
    """

    def __init__(self, size, area, ratio, interp=2, **kwargs):
        super(RandomSizedCropAug, self).__init__(size=size, area=area, ratio=ratio, interp=interp)
        self.size = size
        if 'min_area' in kwargs:
            warnings.warn('`min_area` is deprecated. Please use `area` instead.', DeprecationWarning)
            self.area = kwargs.pop('min_area')
        else:
            self.area = area
        self.ratio = ratio
        self.interp = interp
        assert not kwargs, 'unexpected keyword arguments for `RandomSizedCropAug`.'

    def __call__(self, src):
        """Augmenter body"""
        return random_size_crop(src, self.size, self.area, self.ratio, self.interp)[0]