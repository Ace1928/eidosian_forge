import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _get_fields_and_offsets(dt, offset=0):
    """
    Returns a flat list of (dtype, count, offset) tuples of all the
    scalar fields in the dtype "dt", including nested fields, in left
    to right order.
    """

    def count_elem(dt):
        count = 1
        while dt.shape != ():
            for size in dt.shape:
                count *= size
            dt = dt.base
        return (dt, count)
    fields = []
    for name in dt.names:
        field = dt.fields[name]
        f_dt, f_offset = (field[0], field[1])
        f_dt, n = count_elem(f_dt)
        if f_dt.names is None:
            fields.append((np.dtype((f_dt, (n,))), n, f_offset + offset))
        else:
            subfields = _get_fields_and_offsets(f_dt, f_offset + offset)
            size = f_dt.itemsize
            for i in range(n):
                if i == 0:
                    fields.extend(subfields)
                else:
                    fields.extend([(d, c, o + i * size) for d, c, o in subfields])
    return fields