import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def test_subclass_repr(self):
    """test that repr uses the name of the subclass
        and 'array' for np.ndarray"""
    x = np.arange(5)
    mx = masked_array(x, mask=[True, False, True, False, False])
    assert_startswith(repr(mx), 'masked_array')
    xsub = SubArray(x)
    mxsub = masked_array(xsub, mask=[True, False, True, False, False])
    assert_startswith(repr(mxsub), f'masked_{SubArray.__name__}(data=[--, 1, --, 3, 4]')