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
def write_element(self, arr, mdtype=None):
    """ write tag and data """
    if mdtype is None:
        mdtype = NP_TO_MTYPES[arr.dtype.str[1:]]
    if arr.dtype.byteorder == swapped_code:
        arr = arr.byteswap().view(arr.dtype.newbyteorder())
    byte_count = arr.size * arr.itemsize
    if byte_count <= 4:
        self.write_smalldata_element(arr, mdtype, byte_count)
    else:
        self.write_regular_element(arr, mdtype, byte_count)