from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_pass_timedeltaindex_to_index(self):
    rng = timedelta_range('1 days', '10 days')
    idx = Index(rng, dtype=object)
    expected = Index(rng.to_pytimedelta(), dtype=object)
    tm.assert_numpy_array_equal(idx.values, expected.values)