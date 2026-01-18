import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_axis_mapper(self):
    mi = MultiIndex.from_product([['a', 'b', 'c'], [1, 2]], names=['ll', 'nn'])
    df = DataFrame({'x': list(range(len(mi))), 'y': [i * 10 for i in range(len(mi))]}, index=mi)
    result = df.rename_axis('cols', axis=1)
    tm.assert_index_equal(result.columns, Index(['x', 'y'], name='cols'))
    result = result.rename_axis(columns={'cols': 'new'}, axis=1)
    tm.assert_index_equal(result.columns, Index(['x', 'y'], name='new'))
    result = df.rename_axis(index={'ll': 'foo'})
    assert result.index.names == ['foo', 'nn']
    result = df.rename_axis(index=str.upper, axis=0)
    assert result.index.names == ['LL', 'NN']
    result = df.rename_axis(index=['foo', 'goo'])
    assert result.index.names == ['foo', 'goo']
    sdf = df.reset_index().set_index('nn').drop(columns=['ll', 'y'])
    result = sdf.rename_axis(index='foo', columns='meh')
    assert result.index.name == 'foo'
    assert result.columns.name == 'meh'
    with pytest.raises(TypeError, match='Must pass'):
        df.rename_axis(index='wrong')
    with pytest.raises(ValueError, match='Length of names'):
        df.rename_axis(index=['wrong'])
    with pytest.raises(TypeError, match='bogus'):
        df.rename_axis(bogus=None)