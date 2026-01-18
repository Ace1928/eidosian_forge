from statsmodels.compat.platform import PLATFORM_WIN32
import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.estimators.yule_walker import yule_walker
from statsmodels.tsa.arima.estimators.burg import burg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import (
from statsmodels.tsa.arima.estimators.statespace import statespace
@pytest.mark.parametrize('random_state_type', [7, np.random.RandomState, np.random.default_rng])
def test_reproducible_simulation(random_state_type):
    x = np.random.randn(100)
    res = ARIMA(x, order=(1, 0, 0)).fit()

    def get_random_state(val):
        if isinstance(random_state_type, int):
            return 7
        return random_state_type(7)
    random_state = get_random_state(random_state_type)
    sim1 = res.simulate(1, random_state=random_state)
    random_state = get_random_state(random_state_type)
    sim2 = res.simulate(1, random_state=random_state)
    assert_allclose(sim1, sim2)