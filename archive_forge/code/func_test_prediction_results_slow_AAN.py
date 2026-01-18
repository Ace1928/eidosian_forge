from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
@pytest.mark.skip
def test_prediction_results_slow_AAN(oildata):
    fit = ETSModel(oildata, error='add', trend='add').fit(disp=False)
    pred_exact = fit.get_prediction(start=40, end=55)
    summary_exact = pred_exact.summary_frame()
    pred_sim = fit.get_prediction(start=40, end=55, simulate_repetitions=int(1000000.0), random_state=11, method='simulated')
    summary_sim = pred_sim.summary_frame()
    assert_allclose(summary_sim['mean'].values, summary_sim['mean_numerical'].values, rtol=0.001, atol=0.001)
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
    for i in range(1000):
        plt.plot(pred_sim._results.simulation_results.iloc[:, i], color='grey', alpha=0.1)
    plt.plot(oildata[40:], '-', label='data')
    plt.plot(summary_exact['mean'], '--', label='mean')
    plt.plot(summary_sim['pi_lower'], ':', label='sim lower')
    plt.plot(summary_exact['pi_lower'], '.-', label='exact lower')
    plt.plot(summary_sim['pi_upper'], ':', label='sim upper')
    plt.plot(summary_exact['pi_upper'], '.-', label='exact upper')
    plt.show()
    assert_allclose(summary_sim['pi_lower'].values, summary_exact['pi_lower'].values, rtol=0.0001, atol=0.0001)
    assert_allclose(summary_sim['pi_upper'].values, summary_exact['pi_upper'].values, rtol=0.0001, atol=0.0001)