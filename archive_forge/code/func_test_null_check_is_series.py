from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('null_func', [notna, notnull, isna, isnull])
@pytest.mark.parametrize('ser', [Series([str(i) for i in range(5)], index=Index([str(i) for i in range(5)], dtype=object), dtype=object), Series(range(5), date_range('2020-01-01', periods=5)), Series(range(5), period_range('2020-01-01', periods=5))])
def test_null_check_is_series(null_func, ser):
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with cf.option_context('mode.use_inf_as_na', False):
            assert isinstance(null_func(ser), Series)