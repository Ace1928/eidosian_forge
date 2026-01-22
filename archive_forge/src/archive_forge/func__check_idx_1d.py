from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _check_idx_1d(idx, silent=False):
    if not _is_1d(idx) and np.prod(idx.shape) != np.max(idx.shape):
        if silent:
            return False
        else:
            raise ValueError('Expected idx to be 1D. Got shape {}'.format(idx.shape))
    else:
        return True