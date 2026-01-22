from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _split_rect(x, y, width, height, proportion, horizontal=True, gap=0.05):
    """
    Split the given rectangle in n segments whose proportion is specified
    along the given axis if a gap is inserted, they will be separated by a
    certain amount of space, retaining the relative proportion between them
    a gap of 1 correspond to a plot that is half void and the remaining half
    space is proportionally divided among the pieces.
    """
    x, y, w, h = (float(x), float(y), float(width), float(height))
    if w < 0 or h < 0:
        raise ValueError('dimension of the square less thanzero w={} h={}'.format(w, h))
    proportions = _normalize_split(proportion)
    starting = proportions[:-1]
    amplitude = proportions[1:] - starting
    starting += gap * np.arange(len(proportions) - 1)
    extension = starting[-1] + amplitude[-1] - starting[0]
    starting /= extension
    amplitude /= extension
    starting = (x if horizontal else y) + starting * (w if horizontal else h)
    amplitude = amplitude * (w if horizontal else h)
    results = [(s, y, a, h) if horizontal else (x, s, w, a) for s, a in zip(starting, amplitude)]
    return results