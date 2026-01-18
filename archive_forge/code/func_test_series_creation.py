import os
import subprocess
import sys
import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
import pandas._testing as tm
from pandas.core.internals import (
def test_series_creation():
    msg = 'data_manager option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.data_manager', 'block'):
            s_block = pd.Series([1, 2, 3], name='A', index=['a', 'b', 'c'])
    assert isinstance(s_block._mgr, SingleBlockManager)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.data_manager', 'array'):
            s_array = pd.Series([1, 2, 3], name='A', index=['a', 'b', 'c'])
    assert isinstance(s_array._mgr, SingleArrayManager)
    tm.assert_series_equal(s_block, s_array)
    result = s_block._as_manager('block')
    assert isinstance(result._mgr, SingleBlockManager)
    result = s_block._as_manager('array')
    assert isinstance(result._mgr, SingleArrayManager)
    tm.assert_series_equal(result, s_block)
    result = s_array._as_manager('array')
    assert isinstance(result._mgr, SingleArrayManager)
    result = s_array._as_manager('block')
    assert isinstance(result._mgr, SingleBlockManager)
    tm.assert_series_equal(result, s_array)