import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_spike(self):
    x = np.linspace(0, 10, 1001)
    y = np.cos(x ** 2 / 5)
    result = lowess(y, x, frac=11 / len(x), it=1)
    assert_(np.all(result[:, 1] > np.min(y) - 0.1))
    assert_(np.all(result[:, 1] < np.max(y) + 0.1))