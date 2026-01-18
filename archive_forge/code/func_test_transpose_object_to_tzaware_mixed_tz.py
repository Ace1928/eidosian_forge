import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_transpose_object_to_tzaware_mixed_tz(self):
    dti = date_range('2016-04-05 04:30', periods=3, tz='UTC')
    dti2 = dti.tz_convert('US/Pacific')
    df2 = DataFrame([dti, dti2])
    assert (df2.dtypes == object).all()
    res2 = df2.T
    assert (res2.dtypes == object).all()