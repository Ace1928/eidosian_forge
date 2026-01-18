from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def test_squezable_to_1d():
    data = co2.load().data
    data = data.resample(MONTH_END).mean().ffill()
    res = STL(data).fit()
    assert isinstance(res, DecomposeResult)