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
class ColorJitterAug(RandomOrderAug):
    """Apply random brightness, contrast and saturation jitter in random order.

    Parameters
    ----------
    brightness : float
        The brightness jitter ratio range, [0, 1]
    contrast : float
        The contrast jitter ratio range, [0, 1]
    saturation : float
        The saturation jitter ratio range, [0, 1]
    """

    def __init__(self, brightness, contrast, saturation):
        ts = []
        if brightness > 0:
            ts.append(BrightnessJitterAug(brightness))
        if contrast > 0:
            ts.append(ContrastJitterAug(contrast))
        if saturation > 0:
            ts.append(SaturationJitterAug(saturation))
        super(ColorJitterAug, self).__init__(ts)