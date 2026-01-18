import pickle
import os
import tempfile
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, varmax,
from numpy.testing import assert_allclose
def test_varmax_pickle(temp_filename):
    mod = varmax.VARMAX(macrodata[['realgdp', 'realcons']].diff().iloc[1:].values, order=(1, 0))
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save(temp_filename)
    res2 = varmax.VARMAXResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)