from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
@pytest.mark.smoke
def test_time_varying(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    mod.simulate([], 10)