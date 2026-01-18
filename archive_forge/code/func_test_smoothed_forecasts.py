import numpy as np
import pandas as pd
import os
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose, assert_raises
def test_smoothed_forecasts(self):
    assert_allclose(self.results.smoothed_forecasts.T, self.desired[['muhat1', 'muhat2', 'muhat3']])