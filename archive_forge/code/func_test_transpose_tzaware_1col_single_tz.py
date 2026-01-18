import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_transpose_tzaware_1col_single_tz(self):
    dti = date_range('2016-04-05 04:30', periods=3, tz='UTC')
    df = DataFrame(dti)
    assert (df.dtypes == dti.dtype).all()
    res = df.T
    assert (res.dtypes == dti.dtype).all()