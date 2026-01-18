import numpy as np
import pandas as pd
import os
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose, assert_raises
def test_smoothed_states(self):
    assert_allclose(self.res.smoothed_state[0], self.res_desired.smoothed_state[0])