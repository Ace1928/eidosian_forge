from datetime import timezone
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('copy', [True, False])
def test_tz_localize_copy_inplace_mutate(self, copy, frame_or_series):
    obj = frame_or_series(np.arange(0, 5), index=date_range('20131027', periods=5, freq='1h', tz=None))
    orig = obj.copy()
    result = obj.tz_localize('UTC', copy=copy)
    expected = frame_or_series(np.arange(0, 5), index=date_range('20131027', periods=5, freq='1h', tz='UTC'))
    tm.assert_equal(result, expected)
    tm.assert_equal(obj, orig)
    assert result.index is not obj.index
    assert result is not obj