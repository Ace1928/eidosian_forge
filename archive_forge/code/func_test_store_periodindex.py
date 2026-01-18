import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('format', ['fixed', 'table'])
def test_store_periodindex(tmp_path, setup_path, format):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 1)), index=pd.period_range('20220101', freq='M', periods=5))
    path = tmp_path / setup_path
    df.to_hdf(path, key='df', mode='w', format=format)
    expected = pd.read_hdf(path, 'df')
    tm.assert_frame_equal(df, expected)