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
def write_char(self, arr, codec='ascii'):
    """ Write string array `arr` with given `codec`
        """
    if arr.size == 0 or np.all(arr == ''):
        shape = (0,) * np.max([arr.ndim, 2])
        self.write_header(shape, mxCHAR_CLASS)
        self.write_smalldata_element(arr, miUTF8, 0)
        return
    arr = arr_to_chars(arr)
    shape = arr.shape
    self.write_header(shape, mxCHAR_CLASS)
    if arr.dtype.kind == 'U' and arr.size:
        n_chars = np.prod(shape)
        st_arr = np.ndarray(shape=(), dtype=arr_dtype_number(arr, n_chars), buffer=arr.T.copy())
        st = st_arr.item().encode(codec)
        arr = np.ndarray(shape=(len(st),), dtype='S1', buffer=st)
    self.write_element(arr, mdtype=miUTF8)