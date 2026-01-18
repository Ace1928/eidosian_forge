from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
def test_range_index_columns(self):
    df = pd.DataFrame(np.arange(200).reshape((-1, 2)))
    df.columns = pd.RangeIndex(2)
    result = stattools.lagmat(df, maxlag=2, use_pandas=True)
    assert result.shape == (100, 4)
    assert list(result.columns) == ['0.L.1', '1.L.1', '0.L.2', '1.L.2']