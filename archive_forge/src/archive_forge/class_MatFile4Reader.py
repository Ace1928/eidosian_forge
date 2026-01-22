import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
class MatFile4Reader(MatFileReader):
    """ Reader for Mat4 files """

    @docfiller
    def __init__(self, mat_stream, *args, **kwargs):
        """ Initialize matlab 4 file reader

    %(matstream_arg)s
    %(load_args)s
        """
        super().__init__(mat_stream, *args, **kwargs)
        self._matrix_reader = None

    def guess_byte_order(self):
        self.mat_stream.seek(0)
        mopt = read_dtype(self.mat_stream, np.dtype('i4'))
        self.mat_stream.seek(0)
        if mopt == 0:
            return '<'
        if mopt < 0 or mopt > 5000:
            return SYS_LITTLE_ENDIAN and '>' or '<'
        return SYS_LITTLE_ENDIAN and '<' or '>'

    def initialize_read(self):
        """ Run when beginning read of variables

        Sets up readers from parameters in `self`
        """
        self.dtypes = convert_dtypes(mdtypes_template, self.byte_order)
        self._matrix_reader = VarReader4(self)

    def read_var_header(self):
        """ Read and return header, next position

        Parameters
        ----------
        None

        Returns
        -------
        header : object
           object that can be passed to self.read_var_array, and that
           has attributes ``name`` and ``is_global``
        next_position : int
           position in stream of next variable
        """
        hdr = self._matrix_reader.read_header()
        n = reduce(lambda x, y: x * y, hdr.dims, 1)
        remaining_bytes = hdr.dtype.itemsize * n
        if hdr.is_complex and (not hdr.mclass == mxSPARSE_CLASS):
            remaining_bytes *= 2
        next_position = self.mat_stream.tell() + remaining_bytes
        return (hdr, next_position)

    def read_var_array(self, header, process=True):
        """ Read array, given `header`

        Parameters
        ----------
        header : header object
           object with fields defining variable header
        process : {True, False}, optional
           If True, apply recursive post-processing during loading of array.

        Returns
        -------
        arr : array
           array with post-processing applied or not according to
           `process`.
        """
        return self._matrix_reader.array_from_header(header, process)

    def get_variables(self, variable_names=None):
        """ get variables from stream as dictionary

        Parameters
        ----------
        variable_names : None or str or sequence of str, optional
            variable name, or sequence of variable names to get from Mat file /
            file stream. If None, then get all variables in file.
        """
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        elif variable_names is not None:
            variable_names = list(variable_names)
        self.mat_stream.seek(0)
        self.initialize_read()
        mdict = {}
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            mdict[name] = self.read_var_array(hdr)
            self.mat_stream.seek(next_position)
            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break
        return mdict

    def list_variables(self):
        """ list variables from stream """
        self.mat_stream.seek(0)
        self.initialize_read()
        vars = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            shape = self._matrix_reader.shape_from_header(hdr)
            info = mclass_info.get(hdr.mclass, 'unknown')
            vars.append((name, shape, info))
            self.mat_stream.seek(next_position)
        return vars