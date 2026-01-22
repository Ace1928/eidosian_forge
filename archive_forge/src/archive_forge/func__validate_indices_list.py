import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
def _validate_indices_list(indices_list, formatted):
    prev_ind = None
    for ind in indices_list:
        dims = formatted.annotations['tensor_metadata']['shape']
        if len(ind) != len(dims):
            raise ValueError('Dimensions mismatch: requested: %d; actual: %d' % (len(ind), len(dims)))
        for req_idx, siz in zip(ind, dims):
            if req_idx >= siz:
                raise ValueError('Indices exceed tensor dimensions.')
            if req_idx < 0:
                raise ValueError('Indices contain negative value(s).')
        if prev_ind and ind < prev_ind:
            raise ValueError('Input indices sets are not in ascending order.')
        prev_ind = ind