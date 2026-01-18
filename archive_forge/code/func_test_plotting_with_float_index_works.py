import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
def test_plotting_with_float_index_works(self):
    df = DataFrame({'def': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'val': np.random.default_rng(2).standard_normal(9)}, index=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    df.groupby('def')['val'].plot()