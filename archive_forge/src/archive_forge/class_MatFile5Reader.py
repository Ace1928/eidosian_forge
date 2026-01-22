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
class MatFile5Reader(MatFileReader):
    """ Reader for Mat 5 mat files
    Adds the following attribute to base class

    uint16_codec - char codec to use for uint16 char arrays
        (defaults to system default codec)

    Uses variable reader that has the following standard interface (see
    abstract class in ``miobase``::

       __init__(self, file_reader)
       read_header(self)
       array_from_header(self)

    and added interface::

       set_stream(self, stream)
       read_full_tag(self)

    """

    @docfiller
    def __init__(self, mat_stream, byte_order=None, mat_dtype=False, squeeze_me=False, chars_as_strings=True, matlab_compatible=False, struct_as_record=True, verify_compressed_data_integrity=True, uint16_codec=None, simplify_cells=False):
        """Initializer for matlab 5 file format reader

    %(matstream_arg)s
    %(load_args)s
    %(struct_arg)s
    uint16_codec : {None, string}
        Set codec to use for uint16 char arrays (e.g., 'utf-8').
        Use system default codec if None
        """
        super().__init__(mat_stream, byte_order, mat_dtype, squeeze_me, chars_as_strings, matlab_compatible, struct_as_record, verify_compressed_data_integrity, simplify_cells)
        if not uint16_codec:
            uint16_codec = sys.getdefaultencoding()
        self.uint16_codec = uint16_codec
        self._file_reader = None
        self._matrix_reader = None

    def guess_byte_order(self):
        """ Guess byte order.
        Sets stream pointer to 0"""
        self.mat_stream.seek(126)
        mi = self.mat_stream.read(2)
        self.mat_stream.seek(0)
        return mi == b'IM' and '<' or '>'

    def read_file_header(self):
        """ Read in mat 5 file header """
        hdict = {}
        hdr_dtype = MDTYPES[self.byte_order]['dtypes']['file_header']
        hdr = read_dtype(self.mat_stream, hdr_dtype)
        hdict['__header__'] = hdr['description'].item().strip(b' \t\n\x00')
        v_major = hdr['version'] >> 8
        v_minor = hdr['version'] & 255
        hdict['__version__'] = '%d.%d' % (v_major, v_minor)
        return hdict

    def initialize_read(self):
        """ Run when beginning read of variables

        Sets up readers from parameters in `self`
        """
        self._file_reader = VarReader5(self)
        self._matrix_reader = VarReader5(self)

    def read_var_header(self):
        """ Read header, return header, next position

        Header has to define at least .name and .is_global

        Parameters
        ----------
        None

        Returns
        -------
        header : object
           object that can be passed to self.read_var_array, and that
           has attributes .name and .is_global
        next_position : int
           position in stream of next variable
        """
        mdtype, byte_count = self._file_reader.read_full_tag()
        if not byte_count > 0:
            raise ValueError('Did not read any bytes')
        next_pos = self.mat_stream.tell() + byte_count
        if mdtype == miCOMPRESSED:
            stream = ZlibInputStream(self.mat_stream, byte_count)
            self._matrix_reader.set_stream(stream)
            check_stream_limit = self.verify_compressed_data_integrity
            mdtype, byte_count = self._matrix_reader.read_full_tag()
        else:
            check_stream_limit = False
            self._matrix_reader.set_stream(self.mat_stream)
        if not mdtype == miMATRIX:
            raise TypeError('Expecting miMATRIX type here, got %d' % mdtype)
        header = self._matrix_reader.read_header(check_stream_limit)
        return (header, next_pos)

    def read_var_array(self, header, process=True):
        """ Read array, given `header`

        Parameters
        ----------
        header : header object
           object with fields defining variable header
        process : {True, False} bool, optional
           If True, apply recursive post-processing during loading of
           array.

        Returns
        -------
        arr : array
           array with post-processing applied or not according to
           `process`.
        """
        return self._matrix_reader.array_from_header(header, process)

    def get_variables(self, variable_names=None):
        """ get variables from stream as dictionary

        variable_names   - optional list of variable names to get

        If variable_names is None, then get all variables in file
        """
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        elif variable_names is not None:
            variable_names = list(variable_names)
        self.mat_stream.seek(0)
        self.initialize_read()
        mdict = self.read_file_header()
        mdict['__globals__'] = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if name in mdict:
                warnings.warn('Duplicate variable name "%s" in stream - replacing previous with new\nConsider mio5.varmats_from_mat to split file into single variable files' % name, MatReadWarning, stacklevel=2)
            if name == '':
                name = '__function_workspace__'
                process = False
            else:
                process = True
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            try:
                res = self.read_var_array(hdr, process)
            except MatReadError as err:
                warnings.warn(f'Unreadable variable "{name}", because "{err}"', Warning, stacklevel=2)
                res = 'Read error: %s' % err
            self.mat_stream.seek(next_position)
            mdict[name] = res
            if hdr.is_global:
                mdict['__globals__'].append(name)
            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break
        if self.simplify_cells:
            return _simplify_cells(mdict)
        else:
            return mdict

    def list_variables(self):
        """ list variables from stream """
        self.mat_stream.seek(0)
        self.initialize_read()
        self.read_file_header()
        vars = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if name == '':
                name = '__function_workspace__'
            shape = self._matrix_reader.shape_from_header(hdr)
            if hdr.is_logical:
                info = 'logical'
            else:
                info = mclass_info.get(hdr.mclass, 'unknown')
            vars.append((name, shape, info))
            self.mat_stream.seek(next_position)
        return vars