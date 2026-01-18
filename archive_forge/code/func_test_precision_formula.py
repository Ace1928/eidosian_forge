import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
def test_precision_formula(self):
    m = BetaModel.from_formula(self.model, methylation, exog_precision_formula='~ age', link_precision=links.Identity())
    rslt = m.fit()
    assert_close(rslt.params, self.meth_fit.params, 1e-10)
    assert isinstance(rslt.params, pd.Series)
    with pytest.warns(ValueWarning, match='unknown kwargs'):
        BetaModel.from_formula(self.model, methylation, exog_precision_formula='~ age', link_precision=links.Identity(), junk=False)