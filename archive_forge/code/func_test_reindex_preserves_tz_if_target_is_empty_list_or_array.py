from datetime import timedelta
import numpy as np
from pandas import (
import pandas._testing as tm
def test_reindex_preserves_tz_if_target_is_empty_list_or_array(self):
    index = date_range('2013-01-01', periods=3, tz='US/Eastern')
    assert str(index.reindex([])[0].tz) == 'US/Eastern'
    assert str(index.reindex(np.array([]))[0].tz) == 'US/Eastern'