from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _diff_contrast(self, levels):
    nlevels = len(levels)
    contr = np.zeros((nlevels, nlevels - 1))
    int_range = np.arange(1, nlevels)
    upper_int = np.repeat(int_range, int_range)
    row_i, col_i = np.triu_indices(nlevels - 1)
    col_order = np.argsort(col_i)
    contr[row_i[col_order], col_i[col_order]] = (upper_int - nlevels) / float(nlevels)
    lower_int = np.repeat(int_range, int_range[::-1])
    row_i, col_i = np.tril_indices(nlevels - 1)
    col_order = np.argsort(col_i)
    contr[row_i[col_order] + 1, col_i[col_order]] = lower_int / float(nlevels)
    return contr