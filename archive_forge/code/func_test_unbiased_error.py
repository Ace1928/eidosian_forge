import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
def test_unbiased_error():
    endog = np.arange(1000) * 1.0
    with pytest.raises(ValueError, match='Cannot perform third step'):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True)