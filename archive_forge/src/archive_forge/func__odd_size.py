import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def _odd_size(size, require_odd_size):
    if require_odd_size and size % 2 == 0:
        raise ValueError('expected all footprint elements to have odd size')