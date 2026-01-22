from __future__ import print_function
import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term
def _format_float_column(precision, col):
    format_str = '%.' + str(precision) + 'f'
    assert col.ndim == 1
    simple_float_chars = set('+-0123456789.')
    col_strs = np.array([format_str % (x,) for x in col], dtype=object)
    mask = np.array([simple_float_chars.issuperset(col_str) and '.' in col_str for col_str in col_strs])
    mask_idxes = np.nonzero(mask)[0]
    strip_char = '0'
    if np.any(mask):
        while True:
            if np.all([s.endswith(strip_char) for s in col_strs[mask]]):
                for idx in mask_idxes:
                    col_strs[idx] = col_strs[idx][:-1]
            elif strip_char == '0':
                strip_char = '.'
            else:
                break
    return col_strs