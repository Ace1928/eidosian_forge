from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_timezones_fixed_format_series_nonempty(setup_path, tz_aware_fixture):
    dtype = pd.DatetimeTZDtype(tz=tz_aware_fixture)
    with ensure_clean_store(setup_path) as store:
        s = Series([0], dtype=dtype)
        store['s'] = s
        result = store['s']
        tm.assert_series_equal(result, s)