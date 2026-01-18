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
def test_options_and_result_attributes(self):

    def f(x):
        f.calls += 1
        f.feval += np.size(x)
        return self.f2(x)
    f.ref = self.f2.ref
    f.b = self.f2.b
    default_rtol = 1e-12
    default_atol = f.ref * default_rtol
    f.feval, f.calls = (0, 0)
    ref = _tanhsinh(f, 0, f.b)
    assert self.error(ref.integral, f.ref) < ref.error < default_atol
    assert ref.nfev == f.feval
    ref.calls = f.calls
    assert ref.success
    assert ref.status == 0
    f.feval, f.calls = (0, 0)
    maxlevel = ref.maxlevel
    res = _tanhsinh(f, 0, f.b, maxlevel=maxlevel)
    res.calls = f.calls
    assert res == ref
    f.feval, f.calls = (0, 0)
    maxlevel -= 1
    assert maxlevel >= 2
    res = _tanhsinh(f, 0, f.b, maxlevel=maxlevel)
    assert self.error(res.integral, f.ref) < res.error > default_atol
    assert res.nfev == f.feval < ref.nfev
    assert f.calls == ref.calls - 1
    assert not res.success
    assert res.status == _ECONVERR
    ref = res
    ref.calls = f.calls
    f.feval, f.calls = (0, 0)
    atol = np.nextafter(ref.error, np.inf)
    res = _tanhsinh(f, 0, f.b, rtol=0, atol=atol)
    assert res.integral == ref.integral
    assert res.error == ref.error
    assert res.nfev == f.feval == ref.nfev
    assert f.calls == ref.calls
    assert res.success
    assert res.status == 0
    f.feval, f.calls = (0, 0)
    atol = np.nextafter(ref.error, -np.inf)
    res = _tanhsinh(f, 0, f.b, rtol=0, atol=atol)
    assert self.error(res.integral, f.ref) < res.error < atol
    assert res.nfev == f.feval > ref.nfev
    assert f.calls > ref.calls
    assert res.success
    assert res.status == 0
    f.feval, f.calls = (0, 0)
    rtol = np.nextafter(ref.error / ref.integral, np.inf)
    res = _tanhsinh(f, 0, f.b, rtol=rtol)
    assert res.integral == ref.integral
    assert res.error == ref.error
    assert res.nfev == f.feval == ref.nfev
    assert f.calls == ref.calls
    assert res.success
    assert res.status == 0
    f.feval, f.calls = (0, 0)
    rtol = np.nextafter(ref.error / ref.integral, -np.inf)
    res = _tanhsinh(f, 0, f.b, rtol=rtol)
    assert self.error(res.integral, f.ref) / f.ref < res.error / res.integral < rtol
    assert res.nfev == f.feval > ref.nfev
    assert f.calls > ref.calls
    assert res.success
    assert res.status == 0