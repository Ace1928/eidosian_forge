from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
def test_predict_options(self):
    res = self.res
    n = 5
    pr1 = res.predict(which='prob')
    pr0 = res.predict(exog=res.model.exog[:n], which='prob')
    assert_allclose(pr0, pr1[:n], rtol=1e-10)
    fitted1 = res.predict()
    fitted0 = res.predict(exog=res.model.exog[:n])
    assert_allclose(fitted0, fitted1[:n], rtol=1e-10)