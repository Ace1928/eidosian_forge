import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
def test_series_groupby_plotting_nominally_works(self):
    n = 10
    weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
    gender = np.random.default_rng(2).choice(['male', 'female'], size=n)
    weight.groupby(gender).plot()