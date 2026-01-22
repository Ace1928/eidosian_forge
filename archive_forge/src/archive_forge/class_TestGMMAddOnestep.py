from statsmodels.compat.python import lmap
import numpy as np
import pandas
from scipy import stats
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm
from numpy.testing import assert_allclose, assert_equal
class TestGMMAddOnestep(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()
        endog_name = 'docvis'
        exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
        instrument_names = 'income ssiratio'.split() + XLISTEXOG2 + ['const']
        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]
        asarray = lambda x: np.asarray(x, float)
        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])
        cls.bse_tol = [5e-06, 5e-07]
        q_tol = [0.04, 0]
        start = OLS(np.log(endog + 1), exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.NonlinearIVGMM(endog, exog, instrument, moment_exponential_add)
        res0 = mod.fit(start, maxiter=0, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-08, 'disp': 0}, wargs={'centered': False})
        cls.res1 = res0
        from .results_gmm_poisson import results_addonestep as results
        cls.res2 = results