import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_repeat_range(self, tz_naive_fixture):
    rng = date_range('1/1/2000', '1/1/2001')
    result = rng.repeat(5)
    assert result.freq is None
    assert len(result) == 5 * len(rng)