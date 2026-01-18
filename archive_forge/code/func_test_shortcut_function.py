from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
def test_shortcut_function(self):
    res = pairwise_tukeyhsd(self.endog, self.groups, alpha=self.alpha)
    assert_almost_equal(res.confint, self.res.confint, decimal=14)