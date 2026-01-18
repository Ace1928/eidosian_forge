import collections
from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
def test_value_counts_object_inference_deprecated():
    dti = pd.date_range('2016-01-01', periods=3, tz='UTC')
    idx = dti.astype(object)
    msg = 'The behavior of value_counts with object-dtype is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = idx.value_counts()
    exp = dti.value_counts()
    tm.assert_series_equal(res, exp)