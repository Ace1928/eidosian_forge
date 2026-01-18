import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
def test_str_cat_mixed_inputs(index_or_series):
    box = index_or_series
    s = Index(['a', 'b', 'c', 'd'])
    s = s if box == Index else Series(s, index=s)
    t = Series(['A', 'B', 'C', 'D'], index=s.values)
    d = concat([t, Series(s, index=s)], axis=1)
    expected = Index(['aAa', 'bBb', 'cCc', 'dDd'])
    expected = expected if box == Index else Series(expected.values, index=s.values)
    result = s.str.cat(d)
    tm.assert_equal(result, expected)
    result = s.str.cat(d.values)
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)
    t.index = ['b', 'c', 'd', 'a']
    expected = box(['aDa', 'bAb', 'cBc', 'dCd'])
    expected = expected if box == Index else Series(expected.values, index=s.values)
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)
    d.index = ['b', 'c', 'd', 'a']
    expected = box(['aDd', 'bAa', 'cBb', 'dCc'])
    expected = expected if box == Index else Series(expected.values, index=s.values)
    result = s.str.cat(d)
    tm.assert_equal(result, expected)
    rgx = 'If `others` contains arrays or lists \\(or other list-likes.*'
    z = Series(['1', '2', '3'])
    e = concat([z, z], axis=1)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(e.values)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s.values])
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s])
    rgx = 'others must be Series, Index, DataFrame,.*'
    u = Series(['a', np.nan, 'c', None])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, 'u'])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, d])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, d.values])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, [u, d]])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(set(u))
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, set(u)])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(1)
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(iter([t.values, list(s)]))