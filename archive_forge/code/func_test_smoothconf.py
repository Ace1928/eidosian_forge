import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest
from statsmodels.sandbox.nonparametric import kernels
@pytest.mark.xfail(reason='NaN mismatch', raises=AssertionError, strict=True)
def test_smoothconf(self):
    super().test_smoothconf()