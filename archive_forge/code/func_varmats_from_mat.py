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
def varmats_from_mat(file_obj):
    """ Pull variables out of mat 5 file as a sequence of mat file objects

    This can be useful with a difficult mat file, containing unreadable
    variables. This routine pulls the variables out in raw form and puts them,
    unread, back into a file stream for saving or reading. Another use is the
    pathological case where there is more than one variable of the same name in
    the file; this routine returns the duplicates, whereas the standard reader
    will overwrite duplicates in the returned dictionary.

    The file pointer in `file_obj` will be undefined. File pointers for the
    returned file-like objects are set at 0.

    Parameters
    ----------
    file_obj : file-like
        file object containing mat file

    Returns
    -------
    named_mats : list
        list contains tuples of (name, BytesIO) where BytesIO is a file-like
        object containing mat file contents as for a single variable. The
        BytesIO contains a string with the original header and a single var. If
        ``var_file_obj`` is an individual BytesIO instance, then save as a mat
        file with something like ``open('test.mat',
        'wb').write(var_file_obj.read())``

    Examples
    --------
    >>> import scipy.io
    >>> import numpy as np
    >>> from io import BytesIO
    >>> from scipy.io.matlab._mio5 import varmats_from_mat
    >>> mat_fileobj = BytesIO()
    >>> scipy.io.savemat(mat_fileobj, {'b': np.arange(10), 'a': 'a string'})
    >>> varmats = varmats_from_mat(mat_fileobj)
    >>> sorted([name for name, str_obj in varmats])
    ['a', 'b']
    """
    rdr = MatFile5Reader(file_obj)
    file_obj.seek(0)
    hdr_len = MDTYPES[native_code]['dtypes']['file_header'].itemsize
    raw_hdr = file_obj.read(hdr_len)
    file_obj.seek(0)
    rdr.initialize_read()
    rdr.read_file_header()
    next_position = file_obj.tell()
    named_mats = []
    while not rdr.end_of_stream():
        start_position = next_position
        hdr, next_position = rdr.read_var_header()
        name = 'None' if hdr.name is None else hdr.name.decode('latin1')
        file_obj.seek(start_position)
        byte_count = next_position - start_position
        var_str = file_obj.read(byte_count)
        out_obj = BytesIO()
        out_obj.write(raw_hdr)
        out_obj.write(var_str)
        out_obj.seek(0)
        named_mats.append((name, out_obj))
    return named_mats