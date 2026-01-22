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
class DetHorizontalFlipAug(DetAugmenter):
    """Random horizontal flipping.

    Parameters
    ----------
    p : float
        chance [0, 1] to flip
    """

    def __init__(self, p):
        super(DetHorizontalFlipAug, self).__init__(p=p)
        self.p = p

    def __call__(self, src, label):
        """Augmenter implementation"""
        if random.random() < self.p:
            src = nd.flip(src, axis=1)
            self._flip_label(label)
        return (src, label)

    def _flip_label(self, label):
        """Helper function to flip label."""
        tmp = 1.0 - label[:, 1]
        label[:, 1] = 1.0 - label[:, 3]
        label[:, 3] = tmp