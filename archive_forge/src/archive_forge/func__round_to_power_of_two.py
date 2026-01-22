import numpy as np
import scipy.sparse as sps
from warnings import warn
from ._optimize import OptimizeWarning
from scipy.optimize._remove_redundancy import (
from collections import namedtuple
def _round_to_power_of_two(x):
    """
    Round elements of the array to the nearest power of two.
    """
    return 2 ** np.around(np.log2(x))