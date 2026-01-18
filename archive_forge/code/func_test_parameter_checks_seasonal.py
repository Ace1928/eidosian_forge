from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def test_parameter_checks_seasonal(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs['endog']
    period = class_kwargs['period']
    match = 'seasonal must be an odd positive integer >= 3'
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=2)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=-7)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=period, seasonal=13.0)