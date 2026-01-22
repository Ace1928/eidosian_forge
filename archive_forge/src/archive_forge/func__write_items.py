import os
import time
import sys
import zlib
from io import BytesIO
import warnings
import numpy as np
import scipy.sparse
from ._byteordercodes import native_code, swapped_code
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio5_utils import VarReader5
from ._mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
from ._streams import ZlibInputStream
def _write_items(self, arr):
    fieldnames = [f[0] for f in arr.dtype.descr]
    length = max([len(fieldname) for fieldname in fieldnames]) + 1
    max_length = self.long_field_names and 64 or 32
    if length > max_length:
        raise ValueError('Field names are restricted to %d characters' % (max_length - 1))
    self.write_element(np.array([length], dtype='i4'))
    self.write_element(np.array(fieldnames, dtype='S%d' % length), mdtype=miINT8)
    A = np.atleast_2d(arr).flatten('F')
    for el in A:
        for f in fieldnames:
            self.write(el[f])