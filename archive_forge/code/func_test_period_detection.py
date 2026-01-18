from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def test_period_detection(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    mod = STL(**class_kwargs)
    res = mod.fit()
    del class_kwargs['period']
    endog = class_kwargs['endog']
    index = pd.date_range('1-1-1959', periods=348, freq=MONTH_END)
    class_kwargs['endog'] = pd.Series(endog, index=index)
    mod = STL(**class_kwargs)
    res_implicit_period = mod.fit()
    assert_allclose(res.seasonal, res_implicit_period.seasonal)