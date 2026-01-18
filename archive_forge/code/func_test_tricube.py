import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest
from statsmodels.sandbox.nonparametric import kernels
def test_tricube():
    res_kx = [0.0, 0.1669853116259163, 0.5789448302469136, 0.8243179321289062, 0.8641975308641975, 0.8243179321289062, 0.5789448302469136, 0.1669853116259163, 0.0]
    xx = np.linspace(-1, 1, 9)
    kx = kernels.Tricube()(xx)
    assert_allclose(kx, res_kx, rtol=1e-10)