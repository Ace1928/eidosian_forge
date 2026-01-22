import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestDataFrameQueryStrings:

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

    def test_str_list_query_method(self, parser, engine):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=['b'])
        df['strings'] = Series(list('aabbccddee'))
        expect = df[df.strings.isin(['a', 'b'])]
        if parser != 'pandas':
            col = 'strings'
            lst = '["a", "b"]'
            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]
            eq, ne = ('==', '!=')
            ops = 2 * ([eq] + [ne])
            msg = "'(Not)?In' nodes are not implemented"
            for lhs, op, rhs in zip(lhs, ops, rhs):
                ex = f'{lhs} {op} {rhs}'
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(ex, engine=engine, parser=parser)
        else:
            res = df.query('strings == ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            res = df.query('["a", "b"] == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            expect = df[~df.strings.isin(['a', 'b'])]
            res = df.query('strings != ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            res = df.query('["a", "b"] != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

    def test_query_with_string_columns(self, parser, engine):
        df = DataFrame({'a': list('aaaabbbbcccc'), 'b': list('aabbccddeeff'), 'c': np.random.default_rng(2).integers(5, size=12), 'd': np.random.default_rng(2).integers(9, size=12)})
        if parser == 'pandas':
            res = df.query('a in b', parser=parser, engine=engine)
            expec = df[df.a.isin(df.b)]
            tm.assert_frame_equal(res, expec)
            res = df.query('a in b and c < d', parser=parser, engine=engine)
            expec = df[df.a.isin(df.b) & (df.c < df.d)]
            tm.assert_frame_equal(res, expec)
        else:
            msg = "'(Not)?In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query('a in b', parser=parser, engine=engine)
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query('a in b and c < d', parser=parser, engine=engine)

    def test_object_array_eq_ne(self, parser, engine, using_infer_string):
        df = DataFrame({'a': list('aaaabbbbcccc'), 'b': list('aabbccddeeff'), 'c': np.random.default_rng(2).integers(5, size=12), 'd': np.random.default_rng(2).integers(9, size=12)})
        warning = RuntimeWarning if using_infer_string and engine == 'numexpr' else None
        with tm.assert_produces_warning(warning):
            res = df.query('a == b', parser=parser, engine=engine)
        exp = df[df.a == df.b]
        tm.assert_frame_equal(res, exp)
        with tm.assert_produces_warning(warning):
            res = df.query('a != b', parser=parser, engine=engine)
        exp = df[df.a != df.b]
        tm.assert_frame_equal(res, exp)

    def test_query_with_nested_strings(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        events = [f'page {n} {act}' for n in range(1, 4) for act in ['load', 'exit']] * 2
        stamps1 = date_range('2014-01-01 0:00:01', freq='30s', periods=6)
        stamps2 = date_range('2014-02-01 1:00:01', freq='30s', periods=6)
        df = DataFrame({'id': np.arange(1, 7).repeat(2), 'event': events, 'timestamp': stamps1.append(stamps2)})
        expected = df[df.event == '"page 1 load"']
        res = df.query('\'"page 1 load"\' in event', parser=parser, engine=engine)
        tm.assert_frame_equal(expected, res)

    def test_query_with_nested_special_character(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        df = DataFrame({'a': ['a', 'b', 'test & test'], 'b': [1, 2, 3]})
        res = df.query('a == "test & test"', parser=parser, engine=engine)
        expec = df[df.a == 'test & test']
        tm.assert_frame_equal(res, expec)

    @pytest.mark.parametrize('op, func', [['<', operator.lt], ['>', operator.gt], ['<=', operator.le], ['>=', operator.ge]])
    def test_query_lex_compare_strings(self, parser, engine, op, func, using_infer_string):
        a = Series(np.random.default_rng(2).choice(list('abcde'), 20))
        b = Series(np.arange(a.size))
        df = DataFrame({'X': a, 'Y': b})
        warning = RuntimeWarning if using_infer_string and engine == 'numexpr' else None
        with tm.assert_produces_warning(warning):
            res = df.query(f'X {op} "d"', engine=engine, parser=parser)
        expected = df[func(df.X, 'd')]
        tm.assert_frame_equal(res, expected)

    def test_query_single_element_booleans(self, parser, engine):
        columns = ('bid', 'bidsize', 'ask', 'asksize')
        data = np.random.default_rng(2).integers(2, size=(1, len(columns))).astype(bool)
        df = DataFrame(data, columns=columns)
        res = df.query('bid & ask', engine=engine, parser=parser)
        expected = df[df.bid & df.ask]
        tm.assert_frame_equal(res, expected)

    def test_query_string_scalar_variable(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        df = DataFrame({'Symbol': ['BUD US', 'BUD US', 'IBM US', 'IBM US'], 'Price': [109.7, 109.72, 183.3, 183.35]})
        e = df[df.Symbol == 'BUD US']
        symb = 'BUD US'
        r = df.query('Symbol == @symb', parser=parser, engine=engine)
        tm.assert_frame_equal(e, r)

    @pytest.mark.parametrize('in_list', [[None, 'asdf', 'ghjk'], ['asdf', None, 'ghjk'], ['asdf', 'ghjk', None], [None, None, 'asdf'], ['asdf', None, None], [None, None, None]])
    def test_query_string_null_elements(self, in_list):
        parser = 'pandas'
        engine = 'python'
        expected = {i: value for i, value in enumerate(in_list) if value == 'asdf'}
        df_expected = DataFrame({'a': expected}, dtype='string')
        df_expected.index = df_expected.index.astype('int64')
        df = DataFrame({'a': in_list}, dtype='string')
        res1 = df.query("a == 'asdf'", parser=parser, engine=engine)
        res2 = df[df['a'] == 'asdf']
        res3 = df.query("a <= 'asdf'", parser=parser, engine=engine)
        tm.assert_frame_equal(res1, df_expected)
        tm.assert_frame_equal(res1, res2)
        tm.assert_frame_equal(res1, res3)
        tm.assert_frame_equal(res2, res3)