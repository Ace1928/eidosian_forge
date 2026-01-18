import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def test_subnormal_warning():
    """Test that the subnormal is zero warning is not being raised."""
    with np.errstate(all='ignore'):
        ld_ma = _discovered_machar(np.longdouble)
    bytes = np.dtype(np.longdouble).itemsize
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
            ld_ma.smallest_subnormal
            assert len(w) == 0
        elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
            ld_ma.smallest_subnormal
            assert len(w) == 0
        else:
            ld_ma.smallest_subnormal
            assert len(w) == 0