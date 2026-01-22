from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _key_splitting(rect_dict, keys, values, key_subset, horizontal, gap):
    """
    Given a dictionary where each entry  is a rectangle, a list of key and
    value (count of elements in each category) it split each rect accordingly,
    as long as the key start with the tuple key_subset.  The other keys are
    returned without modification.
    """
    result = {}
    L = len(key_subset)
    for name, (x, y, w, h) in rect_dict.items():
        if key_subset == name[:L]:
            divisions = _split_rect(x, y, w, h, values, horizontal, gap)
            for key, rect in zip(keys, divisions):
                result[name + (key,)] = rect
        else:
            result[name] = (x, y, w, h)
    return result