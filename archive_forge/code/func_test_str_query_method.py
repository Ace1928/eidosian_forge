import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_str_query_method(self, parser, engine):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=['b'])
    df['strings'] = Series(list('aabbccddee'))
    expect = df[df.strings == 'a']
    if parser != 'pandas':
        col = 'strings'
        lst = '"a"'
        lhs = [col] * 2 + [lst] * 2
        rhs = lhs[::-1]
        eq, ne = ('==', '!=')
        ops = 2 * ([eq] + [ne])
        msg = "'(Not)?In' nodes are not implemented"
        for lhs, op, rhs in zip(lhs, ops, rhs):
            ex = f'{lhs} {op} {rhs}'
            with pytest.raises(NotImplementedError, match=msg):
                df.query(ex, engine=engine, parser=parser, local_dict={'strings': df.strings})
    else:
        res = df.query('"a" == strings', engine=engine, parser=parser)
        tm.assert_frame_equal(res, expect)
        res = df.query('strings == "a"', engine=engine, parser=parser)
        tm.assert_frame_equal(res, expect)
        tm.assert_frame_equal(res, df[df.strings.isin(['a'])])
        expect = df[df.strings != 'a']
        res = df.query('strings != "a"', engine=engine, parser=parser)
        tm.assert_frame_equal(res, expect)
        res = df.query('"a" != strings', engine=engine, parser=parser)
        tm.assert_frame_equal(res, expect)
        tm.assert_frame_equal(res, df[~df.strings.isin(['a'])])