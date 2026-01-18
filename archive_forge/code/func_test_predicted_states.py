import numpy as np
import pandas as pd
import os
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax
from statsmodels import datasets
from numpy.testing import assert_equal, assert_allclose, assert_raises
def test_predicted_states(self):
    assert_allclose(self.res.predicted_state[0], self.res_desired.predicted_state[0])