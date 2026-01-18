from copy import deepcopy
import inspect
import pydoc
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import option_context
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('allows_duplicate_labels', [True, False, None])
def test_set_flags(self, allows_duplicate_labels, frame_or_series, using_copy_on_write, warn_copy_on_write):
    obj = DataFrame({'A': [1, 2]})
    key = (0, 0)
    if frame_or_series is Series:
        obj = obj['A']
        key = 0
    result = obj.set_flags(allows_duplicate_labels=allows_duplicate_labels)
    if allows_duplicate_labels is None:
        assert result.flags.allows_duplicate_labels is True
    else:
        assert result.flags.allows_duplicate_labels is allows_duplicate_labels
    assert obj is not result
    assert obj.flags.allows_duplicate_labels is True
    if frame_or_series is Series:
        assert np.may_share_memory(obj.values, result.values)
    else:
        assert np.may_share_memory(obj['A'].values, result['A'].values)
    with tm.assert_cow_warning(warn_copy_on_write):
        result.iloc[key] = 0
    if using_copy_on_write:
        assert obj.iloc[key] == 1
    else:
        assert obj.iloc[key] == 0
        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[key] = 1
    result = obj.set_flags(copy=True, allows_duplicate_labels=allows_duplicate_labels)
    result.iloc[key] = 10
    assert obj.iloc[key] == 1