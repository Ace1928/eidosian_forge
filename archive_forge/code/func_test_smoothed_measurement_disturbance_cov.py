import numpy as np
import pandas as pd
import os
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose, assert_raises
def test_smoothed_measurement_disturbance_cov(self):
    assert_allclose(self.res.smoothed_measurement_disturbance_cov, self.res_desired.smoothed_measurement_disturbance_cov)