import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def _get_powerlaw_dataset_csr(num_rows, num_cols, density=0.1, dtype=None):
    """Returns CSRNDArray with powerlaw distribution
    with exponentially increasing number of non zeros in each row.
    Not supported for cases where total_nnz < 2*num_rows. This is because
    the algorithm first tries to ensure that there are rows with no zeros by
    putting non zeros at beginning of each row.
    """
    _validate_csr_generation_inputs(num_rows, num_cols, density, distribution='powerlaw')
    total_nnz = int(num_rows * num_cols * density)
    unused_nnz = total_nnz
    output_arr = np.zeros((num_rows, num_cols), dtype=dtype)
    for row in range(num_rows):
        output_arr[row][0] = 1 + rnd.uniform(0.001, 2)
        unused_nnz = unused_nnz - 1
        if unused_nnz <= 0:
            return mx.nd.array(output_arr).tostype('csr')
    col_max = 2
    for row in range(num_rows):
        col_limit = min(num_cols, col_max)
        if col_limit == num_cols and unused_nnz > col_limit:
            output_arr[row] = 1 + rnd.uniform(0.001, 2)
            unused_nnz = unused_nnz - col_limit + 1
            if unused_nnz <= 0:
                return mx.nd.array(output_arr).tostype('csr')
            else:
                continue
        for col_index in range(1, col_limit):
            output_arr[row][col_index] = 1 + rnd.uniform(0.001, 2)
            unused_nnz = unused_nnz - 1
            if unused_nnz <= 0:
                return mx.nd.array(output_arr).tostype('csr')
        col_max = col_max * 2
    if unused_nnz > 0:
        raise ValueError('not supported for this density: %s for this shape (%s,%s)' % (density, num_rows, num_cols))
    return mx.nd.array(output_arr).tostype('csr')