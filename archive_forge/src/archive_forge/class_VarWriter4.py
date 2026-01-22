import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
class VarWriter4:

    def __init__(self, file_writer):
        self.file_stream = file_writer.file_stream
        self.oned_as = file_writer.oned_as

    def write_bytes(self, arr):
        self.file_stream.write(arr.tobytes(order='F'))

    def write_string(self, s):
        self.file_stream.write(s)

    def write_header(self, name, shape, P=miDOUBLE, T=mxFULL_CLASS, imagf=0):
        """ Write header for given data options

        Parameters
        ----------
        name : str
            name of variable
        shape : sequence
            Shape of array as it will be read in matlab
        P : int, optional
            code for mat4 data type, one of ``miDOUBLE, miSINGLE, miINT32,
            miINT16, miUINT16, miUINT8``
        T : int, optional
            code for mat4 matrix class, one of ``mxFULL_CLASS, mxCHAR_CLASS,
            mxSPARSE_CLASS``
        imagf : int, optional
            flag indicating complex
        """
        header = np.empty((), mdtypes_template['header'])
        M = not SYS_LITTLE_ENDIAN
        O = 0
        header['mopt'] = M * 1000 + O * 100 + P * 10 + T
        header['mrows'] = shape[0]
        header['ncols'] = shape[1]
        header['imagf'] = imagf
        header['namlen'] = len(name) + 1
        self.write_bytes(header)
        data = name + '\x00'
        self.write_string(data.encode('latin1'))

    def write(self, arr, name):
        """ Write matrix `arr`, with name `name`

        Parameters
        ----------
        arr : array_like
           array to write
        name : str
           name in matlab workspace
        """
        if scipy.sparse.issparse(arr):
            self.write_sparse(arr, name)
            return
        arr = np.asarray(arr)
        dt = arr.dtype
        if not dt.isnative:
            arr = arr.astype(dt.newbyteorder('='))
        dtt = dt.type
        if dtt is np.object_:
            raise TypeError('Cannot save object arrays in Mat4')
        elif dtt is np.void:
            raise TypeError('Cannot save void type arrays')
        elif dtt in (np.str_, np.bytes_):
            self.write_char(arr, name)
            return
        self.write_numeric(arr, name)

    def write_numeric(self, arr, name):
        arr = arr_to_2d(arr, self.oned_as)
        imagf = arr.dtype.kind == 'c'
        try:
            P = np_to_mtypes[arr.dtype.str[1:]]
        except KeyError:
            if imagf:
                arr = arr.astype('c128')
            else:
                arr = arr.astype('f8')
            P = miDOUBLE
        self.write_header(name, arr.shape, P=P, T=mxFULL_CLASS, imagf=imagf)
        if imagf:
            self.write_bytes(arr.real)
            self.write_bytes(arr.imag)
        else:
            self.write_bytes(arr)

    def write_char(self, arr, name):
        arr = arr_to_chars(arr)
        arr = arr_to_2d(arr, self.oned_as)
        dims = arr.shape
        self.write_header(name, dims, P=miUINT8, T=mxCHAR_CLASS)
        if arr.dtype.kind == 'U':
            n_chars = np.prod(dims)
            st_arr = np.ndarray(shape=(), dtype=arr_dtype_number(arr, n_chars), buffer=arr)
            st = st_arr.item().encode('latin-1')
            arr = np.ndarray(shape=dims, dtype='S1', buffer=st)
        self.write_bytes(arr)

    def write_sparse(self, arr, name):
        """ Sparse matrices are 2-D

        See docstring for VarReader4.read_sparse_array
        """
        A = arr.tocoo()
        imagf = A.dtype.kind == 'c'
        ijv = np.zeros((A.nnz + 1, 3 + imagf), dtype='f8')
        ijv[:-1, 0] = A.row
        ijv[:-1, 1] = A.col
        ijv[:-1, 0:2] += 1
        if imagf:
            ijv[:-1, 2] = A.data.real
            ijv[:-1, 3] = A.data.imag
        else:
            ijv[:-1, 2] = A.data
        ijv[-1, 0:2] = A.shape
        self.write_header(name, ijv.shape, P=miDOUBLE, T=mxSPARSE_CLASS)
        self.write_bytes(ijv)