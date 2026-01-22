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
class SaturationJitterAug(Augmenter):
    """Random saturation jitter augmentation.

    Parameters
    ----------
    saturation : float
        The saturation jitter ratio range, [0, 1]
    """

    def __init__(self, saturation):
        super(SaturationJitterAug, self).__init__(saturation=saturation)
        self.saturation = saturation
        self.coef = nd.array([[[0.299, 0.587, 0.114]]])

    def __call__(self, src):
        """Augmenter body"""
        alpha = 1.0 + random.uniform(-self.saturation, self.saturation)
        gray = src * self.coef
        gray = nd.sum(gray, axis=2, keepdims=True)
        gray *= 1.0 - alpha
        src *= alpha
        src += gray
        return src