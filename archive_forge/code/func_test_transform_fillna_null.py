import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_transform_fillna_null():
    df = DataFrame({'price': [10, 10, 20, 20, 30, 30], 'color': [10, 10, 20, 20, 30, 30], 'cost': (100, 200, 300, 400, 500, 600)})
    msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pytest.raises(ValueError, match="Must specify a fill 'value' or 'method'"):
            df.groupby(['price']).transform('fillna')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pytest.raises(ValueError, match="Must specify a fill 'value' or 'method'"):
            df.groupby(['price']).fillna()