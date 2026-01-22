from statsmodels.compat.python import lmap
import numpy as np
import pandas
from scipy import stats
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm
from numpy.testing import assert_allclose, assert_equal
class TestGMMMultTwostepDefault(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = 'aget aget2 educyr actlim totchr'.split()
        endog_name = 'docvis'
        exog_names = 'private medicaid'.split() + XLISTEXOG2 + ['const']
        instrument_names = 'income medicaid ssiratio'.split() + XLISTEXOG2 + ['const']
        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]
        asarray = lambda x: np.asarray(x, float)
        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))
        cls.bse_tol = [0.004, 0.0005]
        cls.params_tol = [5e-05, 5e-05]
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(start, maxiter=2, inv_weights=w0inv, optim_method='bfgs', optim_args={'gtol': 1e-08, 'disp': 0})
        cls.res1 = res0
        from .results_gmm_poisson import results_multtwostepdefault as results
        cls.res2 = results