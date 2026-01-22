import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
class VarReader4:
    """ Class to read matlab 4 variables """

    def __init__(self, file_reader):
        self.file_reader = file_reader
        self.mat_stream = file_reader.mat_stream
        self.dtypes = file_reader.dtypes
        self.chars_as_strings = file_reader.chars_as_strings
        self.squeeze_me = file_reader.squeeze_me

    def read_header(self):
        """ Read and return header for variable """
        data = read_dtype(self.mat_stream, self.dtypes['header'])
        name = self.mat_stream.read(int(data['namlen'])).strip(b'\x00')
        if data['mopt'] < 0 or data['mopt'] > 5000:
            raise ValueError('Mat 4 mopt wrong format, byteswapping problem?')
        M, rest = divmod(data['mopt'], 1000)
        if M not in (0, 1):
            warnings.warn("We do not support byte ordering '%s'; returned data may be corrupt" % order_codes[M], UserWarning, stacklevel=3)
        O, rest = divmod(rest, 100)
        if O != 0:
            raise ValueError('O in MOPT integer should be 0, wrong format?')
        P, rest = divmod(rest, 10)
        T = rest
        dims = (data['mrows'], data['ncols'])
        is_complex = data['imagf'] == 1
        dtype = self.dtypes[P]
        return VarHeader4(name, dtype, T, dims, is_complex)

    def array_from_header(self, hdr, process=True):
        mclass = hdr.mclass
        if mclass == mxFULL_CLASS:
            arr = self.read_full_array(hdr)
        elif mclass == mxCHAR_CLASS:
            arr = self.read_char_array(hdr)
            if process and self.chars_as_strings:
                arr = chars_to_strings(arr)
        elif mclass == mxSPARSE_CLASS:
            return self.read_sparse_array(hdr)
        else:
            raise TypeError('No reader for class code %s' % mclass)
        if process and self.squeeze_me:
            return squeeze_element(arr)
        return arr

    def read_sub_array(self, hdr, copy=True):
        """ Mat4 read using header `hdr` dtype and dims

        Parameters
        ----------
        hdr : object
           object with attributes ``dtype``, ``dims``. dtype is assumed to be
           the correct endianness
        copy : bool, optional
           copies array before return if True (default True)
           (buffer is usually read only)

        Returns
        -------
        arr : ndarray
            of dtype given by `hdr` ``dtype`` and shape given by `hdr` ``dims``
        """
        dt = hdr.dtype
        dims = hdr.dims
        num_bytes = dt.itemsize
        for d in dims:
            num_bytes *= d
        buffer = self.mat_stream.read(int(num_bytes))
        if len(buffer) != num_bytes:
            raise ValueError("Not enough bytes to read matrix '%s'; is this a badly-formed file? Consider listing matrices with `whosmat` and loading named matrices with `variable_names` kwarg to `loadmat`" % hdr.name)
        arr = np.ndarray(shape=dims, dtype=dt, buffer=buffer, order='F')
        if copy:
            arr = arr.copy()
        return arr

    def read_full_array(self, hdr):
        """ Full (rather than sparse) matrix getter

        Read matrix (array) can be real or complex

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ndarray
            complex array if ``hdr.is_complex`` is True, otherwise a real
            numeric array
        """
        if hdr.is_complex:
            res = self.read_sub_array(hdr, copy=False)
            res_j = self.read_sub_array(hdr, copy=False)
            return res + res_j * 1j
        return self.read_sub_array(hdr)

    def read_char_array(self, hdr):
        """ latin-1 text matrix (char matrix) reader

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ndarray
            with dtype 'U1', shape given by `hdr` ``dims``
        """
        arr = self.read_sub_array(hdr).astype(np.uint8)
        S = arr.tobytes().decode('latin-1')
        return np.ndarray(shape=hdr.dims, dtype=np.dtype('U1'), buffer=np.array(S)).copy()

    def read_sparse_array(self, hdr):
        """ Read and return sparse matrix type

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ``scipy.sparse.coo_matrix``
            with dtype ``float`` and shape read from the sparse matrix data

        Notes
        -----
        MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where
        N is the number of non-zero values. Column 1 values [0:N] are the
        (1-based) row indices of the each non-zero value, column 2 [0:N] are the
        column indices, column 3 [0:N] are the (real) values. The last values
        [-1,0:2] of the rows, column indices are shape[0] and shape[1]
        respectively of the output matrix. The last value for the values column
        is a padding 0. mrows and ncols values from the header give the shape of
        the stored matrix, here [N+1, 3]. Complex data are saved as a 4 column
        matrix, where the fourth column contains the imaginary component; the
        last value is again 0. Complex sparse data do *not* have the header
        ``imagf`` field set to True; the fact that the data are complex is only
        detectable because there are 4 storage columns.
        """
        res = self.read_sub_array(hdr)
        tmp = res[:-1, :]
        dims = (int(res[-1, 0]), int(res[-1, 1]))
        I = np.ascontiguousarray(tmp[:, 0], dtype='intc')
        J = np.ascontiguousarray(tmp[:, 1], dtype='intc')
        I -= 1
        J -= 1
        if res.shape[1] == 3:
            V = np.ascontiguousarray(tmp[:, 2], dtype='float')
        else:
            V = np.ascontiguousarray(tmp[:, 2], dtype='complex')
            V.imag = tmp[:, 3]
        return scipy.sparse.coo_matrix((V, (I, J)), dims)

    def shape_from_header(self, hdr):
        """Read the shape of the array described by the header.
        The file position after this call is unspecified.
        """
        mclass = hdr.mclass
        if mclass == mxFULL_CLASS:
            shape = tuple(map(int, hdr.dims))
        elif mclass == mxCHAR_CLASS:
            shape = tuple(map(int, hdr.dims))
            if self.chars_as_strings:
                shape = shape[:-1]
        elif mclass == mxSPARSE_CLASS:
            dt = hdr.dtype
            dims = hdr.dims
            if not (len(dims) == 2 and dims[0] >= 1 and (dims[1] >= 1)):
                return ()
            self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
            rows = np.ndarray(shape=(), dtype=dt, buffer=self.mat_stream.read(dt.itemsize))
            self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
            cols = np.ndarray(shape=(), dtype=dt, buffer=self.mat_stream.read(dt.itemsize))
            shape = (int(rows), int(cols))
        else:
            raise TypeError('No reader for class code %s' % mclass)
        if self.squeeze_me:
            shape = tuple([x for x in shape if x != 1])
        return shape