import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
def test_value_counts_default_dropna(self, data):
    if not hasattr(data, 'value_counts'):
        pytest.skip(f'value_counts is not implemented for {type(data)}')
    sig = inspect.signature(data.value_counts)
    kwarg = sig.parameters['dropna']
    assert kwarg.default is True