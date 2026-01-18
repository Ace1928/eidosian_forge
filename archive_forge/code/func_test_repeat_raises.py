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
@pytest.mark.parametrize('repeats, kwargs, error, msg', [(2, {'axis': 1}, ValueError, 'axis'), (-1, {}, ValueError, 'negative'), ([1, 2], {}, ValueError, 'shape'), (2, {'foo': 'bar'}, TypeError, "'foo'")])
def test_repeat_raises(self, data, repeats, kwargs, error, msg, use_numpy):
    with pytest.raises(error, match=msg):
        if use_numpy:
            np.repeat(data, repeats, **kwargs)
        else:
            data.repeat(repeats, **kwargs)