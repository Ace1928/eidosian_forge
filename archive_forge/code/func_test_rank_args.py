from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('grps', [['qux'], ['qux', 'quux']])
@pytest.mark.parametrize('vals', [np.array([2, 2, 8, 2, 6], dtype=dtype) for dtype in ['i8', 'i4', 'i2', 'i1', 'u8', 'u4', 'u2', 'u1', 'f8', 'f4', 'f2']] + [[pd.Timestamp('2018-01-02'), pd.Timestamp('2018-01-02'), pd.Timestamp('2018-01-08'), pd.Timestamp('2018-01-02'), pd.Timestamp('2018-01-06')], [pd.Timestamp('2018-01-02', tz='US/Pacific'), pd.Timestamp('2018-01-02', tz='US/Pacific'), pd.Timestamp('2018-01-08', tz='US/Pacific'), pd.Timestamp('2018-01-02', tz='US/Pacific'), pd.Timestamp('2018-01-06', tz='US/Pacific')], [pd.Timestamp('2018-01-02') - pd.Timestamp(0), pd.Timestamp('2018-01-02') - pd.Timestamp(0), pd.Timestamp('2018-01-08') - pd.Timestamp(0), pd.Timestamp('2018-01-02') - pd.Timestamp(0), pd.Timestamp('2018-01-06') - pd.Timestamp(0)], [pd.Timestamp('2018-01-02').to_period('D'), pd.Timestamp('2018-01-02').to_period('D'), pd.Timestamp('2018-01-08').to_period('D'), pd.Timestamp('2018-01-02').to_period('D'), pd.Timestamp('2018-01-06').to_period('D')]], ids=lambda x: type(x[0]))
@pytest.mark.parametrize('ties_method,ascending,pct,exp', [('average', True, False, [2.0, 2.0, 5.0, 2.0, 4.0]), ('average', True, True, [0.4, 0.4, 1.0, 0.4, 0.8]), ('average', False, False, [4.0, 4.0, 1.0, 4.0, 2.0]), ('average', False, True, [0.8, 0.8, 0.2, 0.8, 0.4]), ('min', True, False, [1.0, 1.0, 5.0, 1.0, 4.0]), ('min', True, True, [0.2, 0.2, 1.0, 0.2, 0.8]), ('min', False, False, [3.0, 3.0, 1.0, 3.0, 2.0]), ('min', False, True, [0.6, 0.6, 0.2, 0.6, 0.4]), ('max', True, False, [3.0, 3.0, 5.0, 3.0, 4.0]), ('max', True, True, [0.6, 0.6, 1.0, 0.6, 0.8]), ('max', False, False, [5.0, 5.0, 1.0, 5.0, 2.0]), ('max', False, True, [1.0, 1.0, 0.2, 1.0, 0.4]), ('first', True, False, [1.0, 2.0, 5.0, 3.0, 4.0]), ('first', True, True, [0.2, 0.4, 1.0, 0.6, 0.8]), ('first', False, False, [3.0, 4.0, 1.0, 5.0, 2.0]), ('first', False, True, [0.6, 0.8, 0.2, 1.0, 0.4]), ('dense', True, False, [1.0, 1.0, 3.0, 1.0, 2.0]), ('dense', True, True, [1.0 / 3.0, 1.0 / 3.0, 3.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]), ('dense', False, False, [3.0, 3.0, 1.0, 3.0, 2.0]), ('dense', False, True, [3.0 / 3.0, 3.0 / 3.0, 1.0 / 3.0, 3.0 / 3.0, 2.0 / 3.0])])
def test_rank_args(grps, vals, ties_method, ascending, pct, exp):
    key = np.repeat(grps, len(vals))
    orig_vals = vals
    vals = list(vals) * len(grps)
    if isinstance(orig_vals, np.ndarray):
        vals = np.array(vals, dtype=orig_vals.dtype)
    df = DataFrame({'key': key, 'val': vals})
    result = df.groupby('key').rank(method=ties_method, ascending=ascending, pct=pct)
    exp_df = DataFrame(exp * len(grps), columns=['val'])
    tm.assert_frame_equal(result, exp_df)