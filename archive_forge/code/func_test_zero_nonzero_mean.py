from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
def test_zero_nonzero_mean(self):
    mean1 = self.endog.mean()
    mean2 = (1 - self.res.predict(which='prob-zero').mean()) * self.res.predict(which='mean-nonzero').mean()
    assert_allclose(mean1, mean2, atol=0.2)