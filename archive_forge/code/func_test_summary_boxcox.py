from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
def test_summary_boxcox(ses):
    mod = ExponentialSmoothing(ses ** 2, use_boxcox=True, initialization_method='heuristic')
    with pytest.raises(ValueError, match='use_boxcox was set at model'):
        mod.fit(use_boxcox=True)
    res = mod.fit()
    summ = str(res.summary())
    assert re.findall('Box-Cox:[\\s]*True', summ)
    assert isinstance(res.summary().as_text(), str)