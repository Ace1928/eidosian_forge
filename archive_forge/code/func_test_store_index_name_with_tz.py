from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_store_index_name_with_tz(setup_path):
    df = DataFrame({'A': [1, 2]})
    df.index = DatetimeIndex([1234567890123456787, 1234567890123456788])
    df.index = df.index.tz_localize('UTC')
    df.index.name = 'foo'
    with ensure_clean_store(setup_path) as store:
        store.put('frame', df, format='table')
        recons = store['frame']
        tm.assert_frame_equal(recons, df)