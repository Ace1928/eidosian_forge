import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
class VarHeader4:
    is_logical = False
    is_global = False

    def __init__(self, name, dtype, mclass, dims, is_complex):
        self.name = name
        self.dtype = dtype
        self.mclass = mclass
        self.dims = dims
        self.is_complex = is_complex