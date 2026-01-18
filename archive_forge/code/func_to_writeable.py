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
def to_writeable(source):
    """ Convert input object ``source`` to something we can write

    Parameters
    ----------
    source : object

    Returns
    -------
    arr : None or ndarray or EmptyStructMarker
        If `source` cannot be converted to something we can write to a matfile,
        return None.  If `source` is equivalent to an empty dictionary, return
        ``EmptyStructMarker``.  Otherwise return `source` converted to an
        ndarray with contents for writing to matfile.
    """
    if isinstance(source, np.ndarray):
        return source
    if source is None:
        return None
    if hasattr(source, '__array__'):
        return np.asarray(source)
    is_mapping = hasattr(source, 'keys') and hasattr(source, 'values') and hasattr(source, 'items')
    if isinstance(source, np.generic):
        pass
    elif not is_mapping and hasattr(source, '__dict__'):
        source = {key: value for key, value in source.__dict__.items() if not key.startswith('_')}
        is_mapping = True
    if is_mapping:
        dtype = []
        values = []
        for field, value in source.items():
            if isinstance(field, str) and field[0] not in '_0123456789':
                dtype.append((str(field), object))
                values.append(value)
        if dtype:
            return np.array([tuple(values)], dtype)
        else:
            return EmptyStructMarker
    try:
        narr = np.asanyarray(source)
    except ValueError:
        narr = np.asanyarray(source, dtype=object)
    if narr.dtype.type in (object, np.object_) and narr.shape == () and (narr == source):
        return None
    return narr