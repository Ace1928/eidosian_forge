import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_view(self, using_array_manager, using_copy_on_write, warn_copy_on_write):
    dm = DataFrame(np.arange(20.0).reshape(4, 5), index=range(4), columns=range(5))
    df_orig = dm.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            dm.xs(2)[:] = 20
        tm.assert_frame_equal(dm, df_orig)
    elif using_array_manager:
        msg = '\\nA value is trying to be set on a copy of a slice from a DataFrame'
        with pytest.raises(SettingWithCopyError, match=msg):
            dm.xs(2)[:] = 20
        assert not (dm.xs(2) == 20).any()
    else:
        with tm.raises_chained_assignment_error():
            dm.xs(2)[:] = 20
        assert (dm.xs(2) == 20).all()