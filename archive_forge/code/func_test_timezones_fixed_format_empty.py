from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_timezones_fixed_format_empty(setup_path, tz_aware_fixture, frame_or_series):
    dtype = pd.DatetimeTZDtype(tz=tz_aware_fixture)
    obj = Series(dtype=dtype, name='A')
    if frame_or_series is DataFrame:
        obj = obj.to_frame()
    with ensure_clean_store(setup_path) as store:
        store['obj'] = obj
        result = store['obj']
        tm.assert_equal(result, obj)