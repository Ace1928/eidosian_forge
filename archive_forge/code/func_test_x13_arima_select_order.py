from statsmodels.compat.pandas import MONTH_END
import pandas as pd
import pytest
from statsmodels.datasets import co2, macrodata
from statsmodels.tsa.x13 import (
def test_x13_arima_select_order(dataset):
    res = x13_arima_select_order(dataset)
    assert isinstance(res.order, tuple)
    assert isinstance(res.sorder, tuple)