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
def _check_valid_label(self, label):
    """Validate label and its shape."""
    if len(label.shape) != 2 or label.shape[1] < 5:
        msg = 'Label with shape (1+, 5+) required, %s received.' % str(label)
        raise RuntimeError(msg)
    valid_label = np.where(np.logical_and(label[:, 0] >= 0, label[:, 3] > label[:, 1], label[:, 4] > label[:, 2]))[0]
    if valid_label.size < 1:
        raise RuntimeError('Invalid label occurs.')