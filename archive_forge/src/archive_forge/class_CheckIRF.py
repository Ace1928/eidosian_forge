from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
class CheckIRF:
    ref = None
    res = None
    irf = None
    k = None

    def test_irf_coefs(self):
        self._check_irfs(self.irf.irfs, self.ref.irf)
        self._check_irfs(self.irf.orth_irfs, self.ref.orth_irf)

    def _check_irfs(self, py_irfs, r_irfs):
        for i, name in enumerate(self.res.names):
            ref_irfs = r_irfs[name].view((float, self.k), type=np.ndarray)
            res_irfs = py_irfs[:, :, i]
            assert_almost_equal(ref_irfs, res_irfs)

    @pytest.mark.matplotlib
    def test_plot_irf(self, close_figures):
        self.irf.plot()
        self.irf.plot(plot_stderr=False)
        self.irf.plot(impulse=0, response=1)
        self.irf.plot(impulse=0)
        self.irf.plot(response=0)
        self.irf.plot(orth=True)
        self.irf.plot(impulse=0, response=1, orth=True)

    @pytest.mark.matplotlib
    def test_plot_cum_effects(self, close_figures):
        self.irf.plot_cum_effects()
        self.irf.plot_cum_effects(plot_stderr=False)
        self.irf.plot_cum_effects(impulse=0, response=1)
        self.irf.plot_cum_effects(orth=True)
        self.irf.plot_cum_effects(impulse=0, response=1, orth=True)

    @pytest.mark.matplotlib
    def test_plot_figsizes(self):
        assert_equal(self.irf.plot().get_size_inches(), (10, 10))
        assert_equal(self.irf.plot(figsize=(14, 10)).get_size_inches(), (14, 10))
        assert_equal(self.irf.plot_cum_effects().get_size_inches(), (10, 10))
        assert_equal(self.irf.plot_cum_effects(figsize=(14, 10)).get_size_inches(), (14, 10))