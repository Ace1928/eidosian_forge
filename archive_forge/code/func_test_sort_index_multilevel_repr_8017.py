import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('gen,extra', [([1.0, 3.0, 2.0, 5.0], 4.0), ([1, 3, 2, 5], 4), ([Timestamp('20130101'), Timestamp('20130103'), Timestamp('20130102'), Timestamp('20130105')], Timestamp('20130104')), (['1one', '3one', '2one', '5one'], '4one')])
def test_sort_index_multilevel_repr_8017(self, gen, extra):
    data = np.random.default_rng(2).standard_normal((3, 4))
    columns = MultiIndex.from_tuples([('red', i) for i in gen])
    df = DataFrame(data, index=list('def'), columns=columns)
    df2 = pd.concat([df, DataFrame('world', index=list('def'), columns=MultiIndex.from_tuples([('red', extra)]))], axis=1)
    assert str(df2).splitlines()[0].split() == ['red']
    result = df.copy().sort_index(axis=1)
    expected = df.iloc[:, [0, 2, 1, 3]]
    tm.assert_frame_equal(result, expected)
    result = df2.sort_index(axis=1)
    expected = df2.iloc[:, [0, 2, 1, 4, 3]]
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result['red', extra] = 'world'
    result = result.sort_index(axis=1)
    tm.assert_frame_equal(result, expected)