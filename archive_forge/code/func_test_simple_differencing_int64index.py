import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_simple_differencing_int64index():
    values = np.log(realgdp_results['value']).values
    endog = pd.Series(values, index=pd.Index(range(len(values))))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), simple_differencing=True)
    assert_(mod._index.equals(endog.index[1:]))