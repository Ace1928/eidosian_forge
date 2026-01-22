from statsmodels.compat.python import lrange, lzip
from itertools import product
import numpy as np
from numpy import array, cumsum, iterable, r_
from pandas import DataFrame
from statsmodels.graphics import utils
def _reduce_dict(count_dict, partial_key):
    """
    Make partial sum on a counter dict.
    Given a match for the beginning of the category, it will sum each value.
    """
    L = len(partial_key)
    count = sum((v for k, v in count_dict.items() if k[:L] == partial_key))
    return count