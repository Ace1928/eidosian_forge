import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
@pytest.mark.parametrize('axis', [None, 0])
@pytest.mark.parametrize('out_dtype', ['c8', 'f4', 'f8', '>f8', 'i8', 'S4'])
@pytest.mark.parametrize('casting', ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
def test_out_and_dtype(self, axis, out_dtype, casting):
    out = np.empty(4, dtype=out_dtype)
    to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))
    if not np.can_cast(to_concat[0], out_dtype, casting=casting):
        with assert_raises(TypeError):
            concatenate(to_concat, out=out, axis=axis, casting=casting)
        with assert_raises(TypeError):
            concatenate(to_concat, dtype=out.dtype, axis=axis, casting=casting)
    else:
        res_out = concatenate(to_concat, out=out, axis=axis, casting=casting)
        res_dtype = concatenate(to_concat, dtype=out.dtype, axis=axis, casting=casting)
        assert res_out is out
        assert_array_equal(out, res_dtype)
        assert res_dtype.dtype == out_dtype
    with assert_raises(TypeError):
        concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)