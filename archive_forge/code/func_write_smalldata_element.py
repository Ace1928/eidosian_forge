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
def write_smalldata_element(self, arr, mdtype, byte_count):
    tag = np.zeros((), NDT_TAG_SMALL)
    tag['byte_count_mdtype'] = (byte_count << 16) + mdtype
    tag['data'] = arr.tobytes(order='F')
    self.write_bytes(tag)