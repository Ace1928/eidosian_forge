import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_out_1d(self):
    for v in (30, 30.0, 31, 31.0):
        x = masked_array(np.arange(v))
        x[:3] = x[-3:] = masked
        out = masked_array(np.ones(()))
        r = median(x, out=out)
        if v == 30:
            assert_equal(out, 14.5)
        else:
            assert_equal(out, 15.0)
        assert_(r is out)
        assert_(type(r) is MaskedArray)