import pickle
import os
import tempfile
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, varmax,
from numpy.testing import assert_allclose
@pytest.mark.parametrize('order', ((4, 1, 0), (0, 1, 4), (0, 2, 0)))
def test_sarimax_save_remove_data(temp_filename, order):
    mod = sarimax.SARIMAX(macrodata['realgdp'].values, order=order)
    res = mod.smooth(mod.start_params)
    res.summary()
    res.save(temp_filename, remove_data=True)
    res2 = sarimax.SARIMAXResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)