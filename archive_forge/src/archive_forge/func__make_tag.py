import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
from pytest import raises as assert_raises
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u
def _make_tag(base_dt, val, mdtype, sde=False):
    """ Makes a simple matlab tag, full or sde """
    base_dt = np.dtype(base_dt)
    bo = boc.to_numpy_code(base_dt.byteorder)
    byte_count = base_dt.itemsize
    if not sde:
        udt = bo + 'u4'
        padding = 8 - byte_count % 8
        all_dt = [('mdtype', udt), ('byte_count', udt), ('val', base_dt)]
        if padding:
            all_dt.append(('padding', 'u1', padding))
    else:
        udt = bo + 'u2'
        padding = 4 - byte_count
        if bo == '<':
            all_dt = [('mdtype', udt), ('byte_count', udt), ('val', base_dt)]
        else:
            all_dt = [('byte_count', udt), ('mdtype', udt), ('val', base_dt)]
        if padding:
            all_dt.append(('padding', 'u1', padding))
    tag = np.zeros((1,), dtype=all_dt)
    tag['mdtype'] = mdtype
    tag['byte_count'] = byte_count
    tag['val'] = val
    return tag