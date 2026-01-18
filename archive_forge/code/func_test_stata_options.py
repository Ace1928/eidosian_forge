import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_stata_options(fsspectest):
    df = DataFrame({'a': [0]})
    df.to_stata('testmem://mockfile', storage_options={'test': 'stata_write'}, write_index=False)
    assert fsspectest.test[0] == 'stata_write'
    out = read_stata('testmem://mockfile', storage_options={'test': 'stata_read'})
    assert fsspectest.test[0] == 'stata_read'
    tm.assert_frame_equal(df, out.astype('int64'))