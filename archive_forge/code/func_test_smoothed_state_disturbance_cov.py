import numpy as np
import pandas as pd
import os
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose, assert_raises
def test_smoothed_state_disturbance_cov(self):
    assert_allclose(self.res.smoothed_state_disturbance_cov[0, 0], self.res_desired.smoothed_state_disturbance_cov[0, 0])
    assert_allclose(self.res.smoothed_state_disturbance[1, 1], 0)