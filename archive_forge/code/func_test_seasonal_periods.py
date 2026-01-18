from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def test_seasonal_periods(austourists):
    model = ETSModel(austourists, error='add', trend='add', seasonal='add')
    assert model.seasonal_periods == 4
    try:
        model = ETSModel(austourists, seasonal='add', seasonal_periods=0)
    except ValueError:
        pass