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
def write_sparse(self, arr):
    """ Sparse matrices are 2D
        """
    A = arr.tocsc()
    A.sort_indices()
    is_complex = A.dtype.kind == 'c'
    is_logical = A.dtype.kind == 'b'
    nz = A.nnz
    self.write_header(matdims(arr, self.oned_as), mxSPARSE_CLASS, is_complex=is_complex, is_logical=is_logical, nzmax=1 if nz == 0 else nz)
    self.write_element(A.indices.astype('i4'))
    self.write_element(A.indptr.astype('i4'))
    self.write_element(A.data.real)
    if is_complex:
        self.write_element(A.data.imag)