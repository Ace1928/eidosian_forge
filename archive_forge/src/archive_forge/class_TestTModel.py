import pytest
import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.miscmodels.tmodel import TLinearModel
class TestTModel(CheckTLinearModelMixin):

    @classmethod
    def setup_class(cls):
        endog = mm.m_marietta
        exog = add_constant(mm.CRSP)
        mod = TLinearModel(endog, exog)
        res = mod.fit(method='bfgs', disp=False)
        modf = TLinearModel.from_formula('price ~ CRSP', data={'price': mm.m_marietta, 'CRSP': mm.CRSP})
        resf = modf.fit(method='bfgs', disp=False)
        from .results_tmodel import res_t_dfest as res2
        cls.res2 = res2
        cls.res1 = res
        cls.resf = resf
        cls.k_extra = 2