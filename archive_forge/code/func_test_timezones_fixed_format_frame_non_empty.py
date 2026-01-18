from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_timezones_fixed_format_frame_non_empty(setup_path):
    with ensure_clean_store(setup_path) as store:
        rng = date_range('1/1/2000', '1/30/2000', tz='US/Eastern')
        rng = rng._with_freq(None)
        df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng)
        store['df'] = df
        result = store['df']
        tm.assert_frame_equal(result, df)
        _maybe_remove(store, 'df')
        df = DataFrame({'A': rng, 'B': rng.tz_convert('UTC').tz_localize(None), 'C': rng.tz_convert('CET'), 'D': range(len(rng))}, index=rng)
        store['df'] = df
        result = store['df']
        tm.assert_frame_equal(result, df)