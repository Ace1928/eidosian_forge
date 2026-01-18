import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def values_and_dtypes():
    """
    Generate value+dtype pairs that generate floating point errors during
    casts.  The invalid casts to integers will generate "invalid" value
    warnings, the float casts all generate "overflow".

    (The Python int/float paths don't need to get tested in all the same
    situations, but it does not hurt.)
    """
    yield param(70000, 'float16', id='int-to-f2')
    yield param('70000', 'float16', id='str-to-f2')
    yield param(70000.0, 'float16', id='float-to-f2')
    yield param(np.longdouble(70000.0), 'float16', id='longdouble-to-f2')
    yield param(np.float64(70000.0), 'float16', id='double-to-f2')
    yield param(np.float32(70000.0), 'float16', id='float-to-f2')
    yield param(10 ** 100, 'float32', id='int-to-f4')
    yield param(1e+100, 'float32', id='float-to-f2')
    yield param(np.longdouble(1e+300), 'float32', id='longdouble-to-f2')
    yield param(np.float64(1e+300), 'float32', id='double-to-f2')
    max_ld = np.finfo(np.longdouble).max
    spacing = np.spacing(np.nextafter(np.finfo('f8').max, 0))
    if max_ld - spacing > np.finfo('f8').max:
        yield param(np.finfo(np.longdouble).max, 'float64', id='longdouble-to-f8')
    yield param(2e+300, 'complex64', id='float-to-c8')
    yield param(2e+300 + 0j, 'complex64', id='complex-to-c8')
    yield param(2e+300j, 'complex64', id='complex-to-c8')
    yield param(np.longdouble(2e+300), 'complex64', id='longdouble-to-c8')
    with np.errstate(over='ignore'):
        for to_dt in np.typecodes['AllInteger']:
            for value in [np.inf, np.nan]:
                for from_dt in np.typecodes['AllFloat']:
                    from_dt = np.dtype(from_dt)
                    from_val = from_dt.type(value)
                    yield param(from_val, to_dt, id=f'{from_val}-to-{to_dt}')