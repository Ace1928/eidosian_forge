import pickle
import os
import tempfile
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, varmax,
from numpy.testing import assert_allclose
def test_sarimax_pickle():
    mod = sarimax.SARIMAX(macrodata['realgdp'].values, order=(4, 1, 0))
    pkl_mod = pickle.loads(pickle.dumps(mod))
    res = mod.smooth(mod.start_params)
    pkl_res = pkl_mod.smooth(mod.start_params)
    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.bse, pkl_res.bse)
    assert_allclose(res.llf, pkl_res.llf)