import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
@td.skip_if_no('numexpr')
class TestDataFrameQueryNumExprPandas:

    @pytest.fixture
    def engine(self):
        return 'numexpr'

    @pytest.fixture
    def parser(self):
        return 'pandas'

    def test_date_query_with_attribute_access(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df['dates1'] = date_range('1/1/2012', periods=5)
        df['dates2'] = date_range('1/1/2013', periods=5)
        df['dates3'] = date_range('1/1/2014', periods=5)
        res = df.query('@df.dates1 < 20130101 < @df.dates3', engine=engine, parser=parser)
        expec = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_no_attribute_access(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df['dates1'] = date_range('1/1/2012', periods=5)
        df['dates2'] = date_range('1/1/2013', periods=5)
        df['dates3'] = date_range('1/1/2014', periods=5)
        res = df.query('dates1 < 20130101 < dates3', engine=engine, parser=parser)
        expec = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates2'] = date_range('1/1/2013', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates3'] = pd.NaT
        res = df.query('dates1 < 20130101 < dates3', engine=engine, parser=parser)
        expec = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        return_value = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res = df.query('index < 20130101 < dates3', engine=engine, parser=parser)
        expec = df[(df.index < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype({0: object})
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res = df.query('index < 20130101 < dates3', engine=engine, parser=parser)
        expec = df[(df.index < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self, engine, parser):
        n = 10
        d = {}
        d['dates1'] = date_range('1/1/2012', periods=n)
        d['dates3'] = date_range('1/1/2014', periods=n)
        df = DataFrame(d)
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
        return_value = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res = df.query('dates1 < 20130101 < dates3', engine=engine, parser=parser)
        expec = df[(df.index.to_series() < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_non_date(self, engine, parser):
        n = 10
        df = DataFrame({'dates': date_range('1/1/2012', periods=n), 'nondate': np.arange(n)})
        result = df.query('dates == nondate', parser=parser, engine=engine)
        assert len(result) == 0
        result = df.query('dates != nondate', parser=parser, engine=engine)
        tm.assert_frame_equal(result, df)
        msg = 'Invalid comparison between dtype=datetime64\\[ns\\] and ndarray'
        for op in ['<', '>', '<=', '>=']:
            with pytest.raises(TypeError, match=msg):
                df.query(f'dates {op} nondate', parser=parser, engine=engine)

    def test_query_syntax_error(self, engine, parser):
        df = DataFrame({'i': range(10), '+': range(3, 13), 'r': range(4, 14)})
        msg = 'invalid syntax'
        with pytest.raises(SyntaxError, match=msg):
            df.query('i - +', engine=engine, parser=parser)

    def test_query_scope(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        df = DataFrame(np.random.default_rng(2).standard_normal((20, 2)), columns=list('ab'))
        a, b = (1, 2)
        res = df.query('a > b', engine=engine, parser=parser)
        expected = df[df.a > df.b]
        tm.assert_frame_equal(res, expected)
        res = df.query('@a > b', engine=engine, parser=parser)
        expected = df[a > df.b]
        tm.assert_frame_equal(res, expected)
        with pytest.raises(UndefinedVariableError, match="local variable 'c' is not defined"):
            df.query('@a > b > @c', engine=engine, parser=parser)
        with pytest.raises(UndefinedVariableError, match="name 'c' is not defined"):
            df.query('@a > b > c', engine=engine, parser=parser)

    def test_query_doesnt_pickup_local(self, engine, parser):
        n = m = 10
        df = DataFrame(np.random.default_rng(2).integers(m, size=(n, 3)), columns=list('abc'))
        with pytest.raises(UndefinedVariableError, match="name 'sin' is not defined"):
            df.query('sin > 5', engine=engine, parser=parser)

    def test_query_builtin(self, engine, parser):
        n = m = 10
        df = DataFrame(np.random.default_rng(2).integers(m, size=(n, 3)), columns=list('abc'))
        df.index.name = 'sin'
        msg = 'Variables in expression.+'
        with pytest.raises(NumExprClobberingError, match=msg):
            df.query('sin > 5', engine=engine, parser=parser)

    def test_query(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['a', 'b', 'c'])
        tm.assert_frame_equal(df.query('a < b', engine=engine, parser=parser), df[df.a < df.b])
        tm.assert_frame_equal(df.query('a + b > b * c', engine=engine, parser=parser), df[df.a + df.b > df.b * df.c])

    def test_query_index_with_name(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).integers(10, size=(10, 3)), index=Index(range(10), name='blob'), columns=['a', 'b', 'c'])
        res = df.query('(blob < 5) & (a < b)', engine=engine, parser=parser)
        expec = df[(df.index < 5) & (df.a < df.b)]
        tm.assert_frame_equal(res, expec)
        res = df.query('blob < b', engine=engine, parser=parser)
        expec = df[df.index < df.b]
        tm.assert_frame_equal(res, expec)

    def test_query_index_without_name(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).integers(10, size=(10, 3)), index=range(10), columns=['a', 'b', 'c'])
        res = df.query('index < b', engine=engine, parser=parser)
        expec = df[df.index < df.b]
        tm.assert_frame_equal(res, expec)
        res = df.query('index < 5', engine=engine, parser=parser)
        expec = df[df.index < 5]
        tm.assert_frame_equal(res, expec)

    def test_nested_scope(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        expected = df[(df > 0) & (df2 > 0)]
        result = df.query('(@df > 0) & (@df2 > 0)', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)
        result = pd.eval('df[df > 0 and df2 > 0]', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)
        result = pd.eval('df[df > 0 and df2 > 0 and df[df > 0] > 0]', engine=engine, parser=parser)
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        tm.assert_frame_equal(result, expected)
        result = pd.eval('df[(df>0) & (df2>0)]', engine=engine, parser=parser)
        expected = df.query('(@df>0) & (@df2>0)', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_nested_raises_on_local_self_reference(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query('df > 0', engine=engine, parser=parser)

    def test_local_syntax(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        df = DataFrame(np.random.default_rng(2).standard_normal((100, 10)), columns=list('abcdefghij'))
        b = 1
        expect = df[df.a < b]
        result = df.query('a < @b', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)
        expect = df[df.a < df.b]
        result = df.query('a < b', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)

    def test_chained_cmp_and_in(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        cols = list('abc')
        df = DataFrame(np.random.default_rng(2).standard_normal((100, len(cols))), columns=cols)
        res = df.query('a < b < c and a not in b not in c', engine=engine, parser=parser)
        ind = (df.a < df.b) & (df.b < df.c) & ~df.b.isin(df.a) & ~df.c.isin(df.b)
        expec = df[ind]
        tm.assert_frame_equal(res, expec)

    def test_local_variable_with_in(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        a = Series(np.random.default_rng(2).integers(3, size=15), name='a')
        b = Series(np.random.default_rng(2).integers(10, size=15), name='b')
        df = DataFrame({'a': a, 'b': b})
        expected = df.loc[(df.b - 1).isin(a)]
        result = df.query('b - 1 in a', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)
        b = Series(np.random.default_rng(2).integers(10, size=15), name='b')
        expected = df.loc[(b - 1).isin(a)]
        result = df.query('@b - 1 in a', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

    def test_at_inside_string(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        c = 1
        df = DataFrame({'a': ['a', 'a', 'b', 'b', '@c', '@c']})
        result = df.query('a == "@c"', engine=engine, parser=parser)
        expected = df[df.a == '@c']
        tm.assert_frame_equal(result, expected)

    def test_query_undefined_local(self):
        engine, parser = (self.engine, self.parser)
        skip_if_no_pandas_parser(parser)
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list('ab'))
        with pytest.raises(UndefinedVariableError, match="local variable 'c' is not defined"):
            df.query('a == @c', engine=engine, parser=parser)

    def test_index_resolvers_come_after_columns_with_the_same_name(self, engine, parser):
        n = 1
        a = np.r_[20:101:20]
        df = DataFrame({'index': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
        df.index.name = 'index'
        result = df.query('index > 5', engine=engine, parser=parser)
        expected = df[df['index'] > 5]
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'index': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
        result = df.query('ilevel_0 > 5', engine=engine, parser=parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'a': a, 'b': np.random.default_rng(2).standard_normal(a.size)})
        df.index.name = 'a'
        result = df.query('a > 5', engine=engine, parser=parser)
        expected = df[df.a > 5]
        tm.assert_frame_equal(result, expected)
        result = df.query('index > 5', engine=engine, parser=parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('op, f', [['==', operator.eq], ['!=', operator.ne]])
    def test_inf(self, op, f, engine, parser):
        n = 10
        df = DataFrame({'a': np.random.default_rng(2).random(n), 'b': np.random.default_rng(2).random(n)})
        df.loc[::2, 0] = np.inf
        q = f'a {op} inf'
        expected = df[f(df.a, np.inf)]
        result = df.query(q, engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_check_tz_aware_index_query(self, tz_aware_fixture):
        tz = tz_aware_fixture
        df_index = date_range(start='2019-01-01', freq='1d', periods=10, tz=tz, name='time')
        expected = DataFrame(index=df_index)
        df = DataFrame(index=df_index)
        result = df.query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)
        expected = DataFrame(df_index)
        result = df.reset_index().query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)

    def test_method_calls_in_query(self, engine, parser):
        n = 10
        df = DataFrame({'a': 2 * np.random.default_rng(2).random(n), 'b': np.random.default_rng(2).random(n)})
        expected = df[df['a'].astype('int') == 0]
        result = df.query("a.astype('int') == 0", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'a': np.where(np.random.default_rng(2).random(n) < 0.5, np.nan, np.random.default_rng(2).standard_normal(n)), 'b': np.random.default_rng(2).standard_normal(n)})
        expected = df[df['a'].notnull()]
        result = df.query('a.notnull()', engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)