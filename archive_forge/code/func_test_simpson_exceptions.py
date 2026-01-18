import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
@pytest.mark.parametrize(('message', 'kwarg_update'), [('x must be strictly increasing', dict(x=[2, 2, 3, 4])), ('x must be strictly increasing', dict(x=[x0, [2, 2, 4, 8]], y=[y0, y0])), ('x must be strictly increasing', dict(x=[x0, x0, x0], y=[y0, y0, y0], axis=0)), ('At least one point is required', dict(x=[], y=[])), ('`axis=4` is not valid for `y` with `y.ndim=1`', dict(axis=4)), ('shape of `x` must be the same as `y` or 1-D', dict(x=np.arange(5))), ('`initial` must either be a scalar or...', dict(initial=np.arange(5))), ('`dx` must either be a scalar or...', dict(x=None, dx=np.arange(5)))])
def test_simpson_exceptions(self, message, kwarg_update):
    kwargs0 = dict(y=self.y0, x=self.x0, dx=None, initial=None, axis=-1)
    with pytest.raises(ValueError, match=message):
        cumulative_simpson(**dict(kwargs0, **kwarg_update))