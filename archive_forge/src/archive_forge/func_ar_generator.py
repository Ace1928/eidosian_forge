import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
def ar_generator(N=512, sigma=1.0):
    taps = np.array([2.7607, -3.8106, 2.6535, -0.9238])
    v = np.random.normal(size=N, scale=sigma ** 0.5)
    u = np.zeros(N)
    P = len(taps)
    for l in range(P):
        u[l] = v[l] + np.dot(u[:l][::-1], taps[:l])
    for l in range(P, N):
        u[l] = v[l] + np.dot(u[l - P:l][::-1], taps)
    return (u, v, taps)