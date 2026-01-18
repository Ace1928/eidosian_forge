import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pandas as pd
import pytest
from statsmodels.sandbox.nonparametric import kernels
@pytest.mark.slow
@pytest.mark.smoke
def test_smoothconf_data(self):
    kern = self.kern
    crit = 1.9599639845400545
    fitted_x = np.array([kern.smoothconf(x, y, xi) for xi in x])