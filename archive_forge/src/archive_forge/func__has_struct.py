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
def _has_struct(elem):
    """Determine if elem is an array and if first array item is a struct."""
    return isinstance(elem, np.ndarray) and elem.size > 0 and (elem.ndim > 0) and isinstance(elem[0], mat_struct)