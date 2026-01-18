import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_varmax_validate():
    endog = np.log(macrodata[['cpi', 'realgdp']]).diff().iloc[1:]
    exog = np.log(macrodata[['realinv']]).diff().iloc[1:]
    mod1 = varmax.VARMAX(endog, order=(1, 0), exog=exog, measurement_error=True)
    constraints = {'intercept.cpi': 0.5, 'intercept.realgdp': 1.1, 'beta.realinv.cpi': 0.2, 'beta.realinv.realgdp': 0.1, 'sqrt.var.cpi': 1.2, 'sqrt.cov.cpi.realgdp': -0.1, 'sqrt.var.realgdp': 2.3, 'measurement_variance.cpi': 0.4, 'measurement_variance.realgdp': 0.4}
    with mod1.fix_params(constraints):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, constraints)
        assert_equal(mod1._fixed_params_index, [0, 1, 6, 7, 8, 9, 10, 11, 12])
        assert_equal(mod1._free_params_index, [2, 3, 4, 5])
    res1 = mod1.fit_constrained(constraints, disp=False)
    assert_(res1._has_fixed_params)
    assert_equal(res1._fixed_params, constraints)
    assert_equal(res1._fixed_params_index, [0, 1, 6, 7, 8, 9, 10, 11, 12])
    assert_equal(res1._free_params_index, [2, 3, 4, 5])
    mod2 = varmax.VARMAX(endog[['cpi']], order=(1, 0), exog=exog, measurement_error=True)
    constraints = {'L1.cpi.cpi': 0.5}
    with mod2.fix_params(constraints):
        assert_(mod2._has_fixed_params)
        assert_equal(mod2._fixed_params, constraints)
        assert_equal(mod2._fixed_params_index, [1])
        assert_equal(mod2._free_params_index, [0, 2, 3, 4])
    mod3 = varmax.VARMAX(endog[['cpi']], order=(2, 0))
    with pytest.raises(ValueError):
        with mod3.fix_params({'L1.cpi.cpi': 0.5}):
            pass
    constraints = {'L1.cpi.cpi': 0.3, 'L2.cpi.cpi': 0.1}
    with mod3.fix_params(constraints):
        assert_(mod3._has_fixed_params)
        assert_equal(mod3._fixed_params, constraints)
        assert_equal(mod3._fixed_params_index, [1, 2])
        assert_equal(mod3._free_params_index, [0, 3])
    res3 = mod3.fit_constrained(constraints, start_params=[0, 1.0], disp=False)
    assert_(res3._has_fixed_params)
    assert_equal(res3._fixed_params, constraints)
    assert_equal(res3._fixed_params_index, [1, 2])
    assert_equal(res3._free_params_index, [0, 3])
    with mod3.fix_params(constraints):
        with mod3.fix_params({'L1.cpi.cpi': -0.3}):
            assert_(mod3._has_fixed_params)
            assert_equal(mod3._fixed_params, {'L1.cpi.cpi': -0.3, 'L2.cpi.cpi': 0.1})
            assert_equal(mod3._fixed_params_index, [1, 2])
            assert_equal(mod3._free_params_index, [0, 3])
    mod4 = varmax.VARMAX(endog, order=(1, 0))
    with pytest.raises(ValueError):
        with mod4.fix_params({'L1.cpi.cpi': 0.3}):
            pass
    constraints = dict([('L1.cpi.cpi', 0.3), ('L1.realgdp.cpi', 0.1), ('L1.cpi.realgdp', -0.05), ('L1.realgdp.realgdp', 0.1)])
    with mod4.fix_params(constraints):
        assert_(mod4._has_fixed_params)
        assert_equal(mod4._fixed_params, constraints)
        assert_equal(mod4._fixed_params_index, [2, 3, 4, 5])
        assert_equal(mod4._free_params_index, [0, 1, 6, 7, 8])
    res4 = mod4.fit_constrained(constraints, disp=False)
    assert_(res4._has_fixed_params)
    assert_equal(res4._fixed_params, constraints)
    assert_equal(res4._fixed_params_index, [2, 3, 4, 5])
    assert_equal(res4._free_params_index, [0, 1, 6, 7, 8])
    mod5 = varmax.VARMAX(endog[['cpi']], order=(1, 0), enforce_stationarity=False)
    with mod5.fix_params({'L1.cpi.cpi': 0.6}):
        assert_(mod5._has_fixed_params)
        assert_equal(mod5._fixed_params, {'L1.cpi.cpi': 0.6})
        assert_equal(mod5._fixed_params_index, [1])
        assert_equal(mod5._free_params_index, [0, 2])
    mod6 = varmax.VARMAX(endog, order=(1, 0), enforce_stationarity=False)
    with mod6.fix_params({'L1.cpi.cpi': 0.6}):
        assert_(mod6._has_fixed_params)
        assert_equal(mod6._fixed_params, {'L1.cpi.cpi': 0.6})
        assert_equal(mod6._fixed_params_index, [2])
        assert_equal(mod6._free_params_index, [0, 1, 3, 4, 5, 6, 7, 8])