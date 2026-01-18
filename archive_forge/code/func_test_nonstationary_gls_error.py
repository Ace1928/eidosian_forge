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
def test_nonstationary_gls_error():
    endog = pd.read_csv(io.StringIO('data\n\n9.112\n9.102\n9.103\n9.099\n9.094\n9.090\n9.108\n9.088\n9.091\n9.083\n9.095\n\n9.090\n9.098\n9.093\n9.087\n9.088\n9.083\n9.095\n9.077\n9.082\n9.082\n9.081\n\n9.081\n9.079\n9.088\n9.096\n9.081\n9.098\n9.081\n9.094\n9.091\n9.095\n9.097\n\n9.108\n9.104\n9.098\n9.085\n9.093\n9.094\n9.092\n9.093\n9.106\n9.097\n9.108\n\n9.100\n9.106\n9.114\n9.111\n9.097\n9.099\n9.108\n9.108\n9.110\n9.101\n9.111\n\n9.114\n9.111\n9.126\n9.124\n9.112\n9.120\n9.142\n9.136\n9.131\n9.106\n9.112\n\n9.119\n9.125\n9.123\n9.138\n9.133\n9.133\n9.137\n9.133\n9.138\n9.136\n9.128\n\n9.127\n9.143\n9.128\n9.135\n9.133\n9.131\n9.136\n9.120\n9.127\n9.130\n9.116\n\n9.132\n9.128\n9.119\n9.119\n9.110\n9.132\n9.130\n9.124\n9.130\n9.135\n9.135\n\n9.119\n9.119\n9.136\n9.126\n9.122\n9.119\n9.123\n9.121\n9.130\n9.121\n9.119\n\n9.106\n9.118\n9.124\n9.121\n9.127\n9.113\n9.118\n9.103\n9.112\n9.110\n9.111\n\n9.108\n9.113\n9.117\n9.111\n9.100\n9.106\n9.109\n9.113\n9.110\n9.101\n9.113\n\n9.111\n9.101\n9.097\n9.102\n9.100\n9.110\n9.110\n9.096\n9.095\n9.090\n9.104\n\n9.097\n9.099\n9.095\n9.096\n9.085\n9.097\n9.098\n9.090\n9.080\n9.093\n9.085\n\n9.075\n9.067\n9.072\n9.062\n9.068\n9.053\n9.051\n9.049\n9.052\n9.059\n9.070\n\n9.058\n9.074\n9.063\n9.057\n9.062\n9.058\n9.049\n9.047\n9.062\n9.052\n9.052\n\n9.044\n9.060\n9.062\n9.055\n9.058\n9.054\n9.044\n9.047\n9.050\n9.048\n9.041\n\n9.055\n9.051\n9.028\n9.030\n9.029\n9.027\n9.016\n9.023\n9.031\n9.042\n9.035\n\n'), index_col=None)
    mod = ARIMA(endog, order=(18, 0, 39), enforce_stationarity=False, enforce_invertibility=False)
    with pytest.raises(ValueError, match='Roots of the autoregressive'):
        mod.fit(method='hannan_rissanen', low_memory=True, cov_type='none')