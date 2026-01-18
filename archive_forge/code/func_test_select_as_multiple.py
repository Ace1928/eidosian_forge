import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
def test_select_as_multiple(setup_path):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    df2 = df1.copy().rename(columns='{}_2'.format)
    df2['foo'] = 'bar'
    with ensure_clean_store(setup_path) as store:
        msg = 'keys must be a list/tuple'
        with pytest.raises(TypeError, match=msg):
            store.select_as_multiple(None, where=['A>0', 'B>0'], selector='df1')
        store.append('df1', df1, data_columns=['A', 'B'])
        store.append('df2', df2)
        with pytest.raises(TypeError, match=msg):
            store.select_as_multiple(None, where=['A>0', 'B>0'], selector='df1')
        with pytest.raises(TypeError, match=msg):
            store.select_as_multiple([None], where=['A>0', 'B>0'], selector='df1')
        msg = "'No object named df3 in the file'"
        with pytest.raises(KeyError, match=msg):
            store.select_as_multiple(['df1', 'df3'], where=['A>0', 'B>0'], selector='df1')
        with pytest.raises(KeyError, match=msg):
            store.select_as_multiple(['df3'], where=['A>0', 'B>0'], selector='df1')
        with pytest.raises(KeyError, match="'No object named df4 in the file'"):
            store.select_as_multiple(['df1', 'df2'], where=['A>0', 'B>0'], selector='df4')
        result = store.select('df1', ['A>0', 'B>0'])
        expected = store.select_as_multiple(['df1'], where=['A>0', 'B>0'], selector='df1')
        tm.assert_frame_equal(result, expected)
        expected = store.select_as_multiple('df1', where=['A>0', 'B>0'], selector='df1')
        tm.assert_frame_equal(result, expected)
        result = store.select_as_multiple(['df1', 'df2'], where=['A>0', 'B>0'], selector='df1')
        expected = concat([df1, df2], axis=1)
        expected = expected[(expected.A > 0) & (expected.B > 0)]
        tm.assert_frame_equal(result, expected, check_freq=False)
        result = store.select_as_multiple(['df1', 'df2'], where='index>df2.index[4]', selector='df2')
        expected = concat([df1, df2], axis=1)
        expected = expected[5:]
        tm.assert_frame_equal(result, expected)
        df3 = df1.copy().head(2)
        store.append('df3', df3)
        msg = 'all tables must have exactly the same nrows!'
        with pytest.raises(ValueError, match=msg):
            store.select_as_multiple(['df1', 'df3'], where=['A>0', 'B>0'], selector='df1')