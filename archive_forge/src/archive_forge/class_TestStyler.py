import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
class TestStyler:

    def test_init_non_pandas(self):
        msg = '``data`` must be a Series or DataFrame'
        with pytest.raises(TypeError, match=msg):
            Styler([1, 2, 3])

    def test_init_series(self):
        result = Styler(Series([1, 2]))
        assert result.data.ndim == 2

    def test_repr_html_ok(self, styler):
        styler._repr_html_()

    def test_repr_html_mathjax(self, styler):
        assert 'tex2jax_ignore' not in styler._repr_html_()
        with option_context('styler.html.mathjax', False):
            assert 'tex2jax_ignore' in styler._repr_html_()

    def test_update_ctx(self, styler):
        styler._update_ctx(DataFrame({'A': ['color: red', 'color: blue']}))
        expected = {(0, 0): [('color', 'red')], (1, 0): [('color', 'blue')]}
        assert styler.ctx == expected

    def test_update_ctx_flatten_multi_and_trailing_semi(self, styler):
        attrs = DataFrame({'A': ['color: red; foo: bar', 'color:blue ; foo: baz;']})
        styler._update_ctx(attrs)
        expected = {(0, 0): [('color', 'red'), ('foo', 'bar')], (1, 0): [('color', 'blue'), ('foo', 'baz')]}
        assert styler.ctx == expected

    def test_render(self):
        df = DataFrame({'A': [0, 1]})
        style = lambda x: Series(['color: red', 'color: blue'], name=x.name)
        s = Styler(df, uuid='AB').apply(style)
        s.to_html()

    def test_multiple_render(self, df):
        s = Styler(df, uuid_len=0).map(lambda x: 'color: red;', subset=['A'])
        s.to_html()
        assert '<style type="text/css">\n#T__row0_col0, #T__row1_col0 {\n  color: red;\n}\n</style>' in s.to_html()

    def test_render_empty_dfs(self):
        empty_df = DataFrame()
        es = Styler(empty_df)
        es.to_html()
        DataFrame(columns=['a']).style.to_html()
        DataFrame(index=['a']).style.to_html()

    def test_render_double(self):
        df = DataFrame({'A': [0, 1]})
        style = lambda x: Series(['color: red; border: 1px', 'color: blue; border: 2px'], name=x.name)
        s = Styler(df, uuid='AB').apply(style)
        s.to_html()

    def test_set_properties(self):
        df = DataFrame({'A': [0, 1]})
        result = df.style.set_properties(color='white', size='10px')._compute().ctx
        v = [('color', 'white'), ('size', '10px')]
        expected = {(0, 0): v, (1, 0): v}
        assert result.keys() == expected.keys()
        for v1, v2 in zip(result.values(), expected.values()):
            assert sorted(v1) == sorted(v2)

    def test_set_properties_subset(self):
        df = DataFrame({'A': [0, 1]})
        result = df.style.set_properties(subset=IndexSlice[0, 'A'], color='white')._compute().ctx
        expected = {(0, 0): [('color', 'white')]}
        assert result == expected

    def test_empty_index_name_doesnt_display(self, blank_value):
        df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = df.style._translate(True, True)
        assert len(result['head']) == 1
        expected = {'class': 'blank level0', 'type': 'th', 'value': blank_value, 'is_visible': True, 'display_value': blank_value}
        assert expected.items() <= result['head'][0][0].items()

    def test_index_name(self):
        df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = df.set_index('A').style._translate(True, True)
        expected = {'class': 'index_name level0', 'type': 'th', 'value': 'A', 'is_visible': True, 'display_value': 'A'}
        assert expected.items() <= result['head'][1][0].items()

    def test_numeric_columns(self):
        df = DataFrame({0: [1, 2, 3]})
        df.style._translate(True, True)

    def test_apply_axis(self):
        df = DataFrame({'A': [0, 0], 'B': [1, 1]})
        f = lambda x: [f'val: {x.max()}' for v in x]
        result = df.style.apply(f, axis=1)
        assert len(result._todo) == 1
        assert len(result.ctx) == 0
        result._compute()
        expected = {(0, 0): [('val', '1')], (0, 1): [('val', '1')], (1, 0): [('val', '1')], (1, 1): [('val', '1')]}
        assert result.ctx == expected
        result = df.style.apply(f, axis=0)
        expected = {(0, 0): [('val', '0')], (0, 1): [('val', '1')], (1, 0): [('val', '0')], (1, 1): [('val', '1')]}
        result._compute()
        assert result.ctx == expected
        result = df.style.apply(f)
        result._compute()
        assert result.ctx == expected

    @pytest.mark.parametrize('axis', [0, 1])
    def test_apply_series_return(self, axis):
        df = DataFrame([[1, 2], [3, 4]], index=['X', 'Y'], columns=['X', 'Y'])
        func = lambda s: Series(['color: red;'], index=['Y'])
        result = df.style.apply(func, axis=axis)._compute().ctx
        assert result[1, 1] == [('color', 'red')]
        assert result[1 - axis, axis] == [('color', 'red')]
        func = lambda s: Series(['color: red;', 'color: blue;'], index=['Y', 'X'])
        result = df.style.apply(func, axis=axis)._compute().ctx
        assert result[0, 0] == [('color', 'blue')]
        assert result[1, 1] == [('color', 'red')]
        assert result[1 - axis, axis] == [('color', 'red')]
        assert result[axis, 1 - axis] == [('color', 'blue')]

    @pytest.mark.parametrize('index', [False, True])
    @pytest.mark.parametrize('columns', [False, True])
    def test_apply_dataframe_return(self, index, columns):
        df = DataFrame([[1, 2], [3, 4]], index=['X', 'Y'], columns=['X', 'Y'])
        idxs = ['X', 'Y'] if index else ['Y']
        cols = ['X', 'Y'] if columns else ['Y']
        df_styles = DataFrame('color: red;', index=idxs, columns=cols)
        result = df.style.apply(lambda x: df_styles, axis=None)._compute().ctx
        assert result[1, 1] == [('color', 'red')]
        assert (result[0, 1] == [('color', 'red')]) is index
        assert (result[1, 0] == [('color', 'red')]) is columns
        assert (result[0, 0] == [('color', 'red')]) is (index and columns)

    @pytest.mark.parametrize('slice_', [IndexSlice[:], IndexSlice[:, ['A']], IndexSlice[[1], :], IndexSlice[[1], ['A']], IndexSlice[:2, ['A', 'B']]])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_apply_subset(self, slice_, axis, df):

        def h(x, color='bar'):
            return Series(f'color: {color}', index=x.index, name=x.name)
        result = df.style.apply(h, axis=axis, subset=slice_, color='baz')._compute().ctx
        expected = {(r, c): [('color', 'baz')] for r, row in enumerate(df.index) for c, col in enumerate(df.columns) if row in df.loc[slice_].index and col in df.loc[slice_].columns}
        assert result == expected

    @pytest.mark.parametrize('slice_', [IndexSlice[:], IndexSlice[:, ['A']], IndexSlice[[1], :], IndexSlice[[1], ['A']], IndexSlice[:2, ['A', 'B']]])
    def test_map_subset(self, slice_, df):
        result = df.style.map(lambda x: 'color:baz;', subset=slice_)._compute().ctx
        expected = {(r, c): [('color', 'baz')] for r, row in enumerate(df.index) for c, col in enumerate(df.columns) if row in df.loc[slice_].index and col in df.loc[slice_].columns}
        assert result == expected

    @pytest.mark.parametrize('slice_', [IndexSlice[:, IndexSlice['x', 'A']], IndexSlice[:, IndexSlice[:, 'A']], IndexSlice[:, IndexSlice[:, ['A', 'C']]], IndexSlice[IndexSlice['a', 1], :], IndexSlice[IndexSlice[:, 1], :], IndexSlice[IndexSlice[:, [1, 3]], :], IndexSlice[:, ('x', 'A')], IndexSlice[('a', 1), :]])
    def test_map_subset_multiindex(self, slice_):
        if isinstance(slice_[-1], tuple) and isinstance(slice_[-1][-1], list) and ('C' in slice_[-1][-1]):
            ctx = pytest.raises(KeyError, match='C')
        elif isinstance(slice_[0], tuple) and isinstance(slice_[0][1], list) and (3 in slice_[0][1]):
            ctx = pytest.raises(KeyError, match='3')
        else:
            ctx = contextlib.nullcontext()
        idx = MultiIndex.from_product([['a', 'b'], [1, 2]])
        col = MultiIndex.from_product([['x', 'y'], ['A', 'B']])
        df = DataFrame(np.random.default_rng(2).random((4, 4)), columns=col, index=idx)
        with ctx:
            df.style.map(lambda x: 'color: red;', subset=slice_).to_html()

    def test_map_subset_multiindex_code(self):
        codes = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        columns = MultiIndex(levels=[['a', 'b'], ['%', '#']], codes=codes, names=['', ''])
        df = DataFrame([[1, -1, 1, 1], [-1, 1, 1, 1]], index=['hello', 'world'], columns=columns)
        pct_subset = IndexSlice[:, IndexSlice[:, '%':'%']]

        def color_negative_red(val):
            color = 'red' if val < 0 else 'black'
            return f'color: {color}'
        df.loc[pct_subset]
        df.style.map(color_negative_red, subset=pct_subset)

    @pytest.mark.parametrize('stylefunc', ['background_gradient', 'bar', 'text_gradient'])
    def test_subset_for_boolean_cols(self, stylefunc):
        df = DataFrame([[1, 2], [3, 4]], columns=[False, True])
        styled = getattr(df.style, stylefunc)()
        styled._compute()
        assert set(styled.ctx) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_empty(self):
        df = DataFrame({'A': [1, 0]})
        s = df.style
        s.ctx = {(0, 0): [('color', 'red')], (1, 0): [('', '')]}
        result = s._translate(True, True)['cellstyle']
        expected = [{'props': [('color', 'red')], 'selectors': ['row0_col0']}, {'props': [('', '')], 'selectors': ['row1_col0']}]
        assert result == expected

    def test_duplicate(self):
        df = DataFrame({'A': [1, 0]})
        s = df.style
        s.ctx = {(0, 0): [('color', 'red')], (1, 0): [('color', 'red')]}
        result = s._translate(True, True)['cellstyle']
        expected = [{'props': [('color', 'red')], 'selectors': ['row0_col0', 'row1_col0']}]
        assert result == expected

    def test_init_with_na_rep(self):
        df = DataFrame([[None, None], [1.1, 1.2]], columns=['A', 'B'])
        ctx = Styler(df, na_rep='NA')._translate(True, True)
        assert ctx['body'][0][1]['display_value'] == 'NA'
        assert ctx['body'][0][2]['display_value'] == 'NA'

    def test_caption(self, df):
        styler = Styler(df, caption='foo')
        result = styler.to_html()
        assert all(['caption' in result, 'foo' in result])
        styler = df.style
        result = styler.set_caption('baz')
        assert styler is result
        assert styler.caption == 'baz'

    def test_uuid(self, df):
        styler = Styler(df, uuid='abc123')
        result = styler.to_html()
        assert 'abc123' in result
        styler = df.style
        result = styler.set_uuid('aaa')
        assert result is styler
        assert result.uuid == 'aaa'

    def test_unique_id(self):
        df = DataFrame({'a': [1, 3, 5, 6], 'b': [2, 4, 12, 21]})
        result = df.style.to_html(uuid='test')
        assert 'test' in result
        ids = re.findall('id="(.*?)"', result)
        assert np.unique(ids).size == len(ids)

    def test_table_styles(self, df):
        style = [{'selector': 'th', 'props': [('foo', 'bar')]}]
        styler = Styler(df, table_styles=style)
        result = ' '.join(styler.to_html().split())
        assert 'th { foo: bar; }' in result
        styler = df.style
        result = styler.set_table_styles(style)
        assert styler is result
        assert styler.table_styles == style
        style = [{'selector': 'th', 'props': 'foo:bar;'}]
        styler = df.style.set_table_styles(style)
        result = ' '.join(styler.to_html().split())
        assert 'th { foo: bar; }' in result

    def test_table_styles_multiple(self, df):
        ctx = df.style.set_table_styles([{'selector': 'th,td', 'props': 'color:red;'}, {'selector': 'tr', 'props': 'color:green;'}])._translate(True, True)['table_styles']
        assert ctx == [{'selector': 'th', 'props': [('color', 'red')]}, {'selector': 'td', 'props': [('color', 'red')]}, {'selector': 'tr', 'props': [('color', 'green')]}]

    def test_table_styles_dict_multiple_selectors(self, df):
        result = df.style.set_table_styles({'B': [{'selector': 'th,td', 'props': [('border-left', '2px solid black')]}]})._translate(True, True)['table_styles']
        expected = [{'selector': 'th.col1', 'props': [('border-left', '2px solid black')]}, {'selector': 'td.col1', 'props': [('border-left', '2px solid black')]}]
        assert result == expected

    def test_maybe_convert_css_to_tuples(self):
        expected = [('a', 'b'), ('c', 'd e')]
        assert maybe_convert_css_to_tuples('a:b;c:d e;') == expected
        assert maybe_convert_css_to_tuples('a: b ;c:  d e  ') == expected
        expected = []
        assert maybe_convert_css_to_tuples('') == expected

    def test_maybe_convert_css_to_tuples_err(self):
        msg = 'Styles supplied as string must follow CSS rule formats'
        with pytest.raises(ValueError, match=msg):
            maybe_convert_css_to_tuples('err')

    def test_table_attributes(self, df):
        attributes = 'class="foo" data-bar'
        styler = Styler(df, table_attributes=attributes)
        result = styler.to_html()
        assert 'class="foo" data-bar' in result
        result = df.style.set_table_attributes(attributes).to_html()
        assert 'class="foo" data-bar' in result

    def test_apply_none(self):

        def f(x):
            return DataFrame(np.where(x == x.max(), 'color: red', ''), index=x.index, columns=x.columns)
        result = DataFrame([[1, 2], [3, 4]]).style.apply(f, axis=None)._compute().ctx
        assert result[1, 1] == [('color', 'red')]

    def test_trim(self, df):
        result = df.style.to_html()
        assert result.count('#') == 0
        result = df.style.highlight_max().to_html()
        assert result.count('#') == len(df.columns)

    def test_export(self, df, styler):
        f = lambda x: 'color: red' if x > 0 else 'color: blue'
        g = lambda x, z: f'color: {z}' if x > 0 else f'color: {z}'
        style1 = styler
        style1.map(f).map(g, z='b').highlight_max()._compute()
        result = style1.export()
        style2 = df.style
        style2.use(result)
        assert style1._todo == style2._todo
        style2.to_html()

    def test_bad_apply_shape(self):
        df = DataFrame([[1, 2], [3, 4]], index=['A', 'B'], columns=['X', 'Y'])
        msg = 'resulted in the apply method collapsing to a Series.'
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: 'x')
        msg = 'created invalid {} labels'
        with pytest.raises(ValueError, match=msg.format('index')):
            df.style._apply(lambda x: [''])
        with pytest.raises(ValueError, match=msg.format('index')):
            df.style._apply(lambda x: ['', '', '', ''])
        with pytest.raises(ValueError, match=msg.format('index')):
            df.style._apply(lambda x: Series(['a:v;', ''], index=['A', 'C']), axis=0)
        with pytest.raises(ValueError, match=msg.format('columns')):
            df.style._apply(lambda x: ['', '', ''], axis=1)
        with pytest.raises(ValueError, match=msg.format('columns')):
            df.style._apply(lambda x: Series(['a:v;', ''], index=['X', 'Z']), axis=1)
        msg = 'returned ndarray with wrong shape'
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: np.array([[''], ['']]), axis=None)

    def test_apply_bad_return(self):

        def f(x):
            return ''
        df = DataFrame([[1, 2], [3, 4]])
        msg = 'must return a DataFrame or ndarray when passed to `Styler.apply` with axis=None'
        with pytest.raises(TypeError, match=msg):
            df.style._apply(f, axis=None)

    @pytest.mark.parametrize('axis', ['index', 'columns'])
    def test_apply_bad_labels(self, axis):

        def f(x):
            return DataFrame(**{axis: ['bad', 'labels']})
        df = DataFrame([[1, 2], [3, 4]])
        msg = f'created invalid {axis} labels.'
        with pytest.raises(ValueError, match=msg):
            df.style._apply(f, axis=None)

    def test_get_level_lengths(self):
        index = MultiIndex.from_product([['a', 'b'], [0, 1, 2]])
        expected = {(0, 0): 3, (0, 3): 3, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1}
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)
        expected = {(0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1, (0, 5): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1}
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_get_level_lengths_un_sorted(self):
        index = MultiIndex.from_arrays([[1, 1, 2, 1], ['a', 'b', 'b', 'd']])
        expected = {(0, 0): 2, (0, 2): 1, (0, 3): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1}
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)
        expected = {(0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1}
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_mi_sparse_index_names(self, blank_value):
        df = DataFrame({'A': [1, 2]}, index=MultiIndex.from_arrays([['a', 'a'], [0, 1]], names=['idx_level_0', 'idx_level_1']))
        result = df.style._translate(True, True)
        head = result['head'][1]
        expected = [{'class': 'index_name level0', 'display_value': 'idx_level_0', 'is_visible': True}, {'class': 'index_name level1', 'display_value': 'idx_level_1', 'is_visible': True}, {'class': 'blank col0', 'display_value': blank_value, 'is_visible': True}]
        for i, expected_dict in enumerate(expected):
            assert expected_dict.items() <= head[i].items()

    def test_mi_sparse_column_names(self, blank_value):
        df = DataFrame(np.arange(16).reshape(4, 4), index=MultiIndex.from_arrays([['a', 'a', 'b', 'a'], [0, 1, 1, 2]], names=['idx_level_0', 'idx_level_1']), columns=MultiIndex.from_arrays([['C1', 'C1', 'C2', 'C2'], [1, 0, 1, 0]], names=['colnam_0', 'colnam_1']))
        result = Styler(df, cell_ids=False)._translate(True, True)
        for level in [0, 1]:
            head = result['head'][level]
            expected = [{'class': 'blank', 'display_value': blank_value, 'is_visible': True}, {'class': f'index_name level{level}', 'display_value': f'colnam_{level}', 'is_visible': True}]
            for i, expected_dict in enumerate(expected):
                assert expected_dict.items() <= head[i].items()

    def test_hide_column_headers(self, df, styler):
        ctx = styler.hide(axis='columns')._translate(True, True)
        assert len(ctx['head']) == 0
        df.index.name = 'some_name'
        ctx = df.style.hide(axis='columns')._translate(True, True)
        assert len(ctx['head']) == 1

    def test_hide_single_index(self, df):
        ctx = df.style._translate(True, True)
        assert ctx['body'][0][0]['is_visible']
        assert ctx['head'][0][0]['is_visible']
        ctx2 = df.style.hide(axis='index')._translate(True, True)
        assert not ctx2['body'][0][0]['is_visible']
        assert not ctx2['head'][0][0]['is_visible']
        ctx3 = df.set_index('A').style._translate(True, True)
        assert ctx3['body'][0][0]['is_visible']
        assert len(ctx3['head']) == 2
        assert ctx3['head'][0][0]['is_visible']
        ctx4 = df.set_index('A').style.hide(axis='index')._translate(True, True)
        assert not ctx4['body'][0][0]['is_visible']
        assert len(ctx4['head']) == 1
        assert not ctx4['head'][0][0]['is_visible']

    def test_hide_multiindex(self):
        df = DataFrame({'A': [1, 2], 'B': [1, 2]}, index=MultiIndex.from_arrays([['a', 'a'], [0, 1]], names=['idx_level_0', 'idx_level_1']))
        ctx1 = df.style._translate(True, True)
        assert ctx1['body'][0][0]['is_visible']
        assert ctx1['body'][0][1]['is_visible']
        assert len(ctx1['head'][0]) == 4
        ctx2 = df.style.hide(axis='index')._translate(True, True)
        assert not ctx2['body'][0][0]['is_visible']
        assert not ctx2['body'][0][1]['is_visible']
        assert len(ctx2['head'][0]) == 3
        assert not ctx2['head'][0][0]['is_visible']

    def test_hide_columns_single_level(self, df):
        ctx = df.style._translate(True, True)
        assert ctx['head'][0][1]['is_visible']
        assert ctx['head'][0][1]['display_value'] == 'A'
        assert ctx['head'][0][2]['is_visible']
        assert ctx['head'][0][2]['display_value'] == 'B'
        assert ctx['body'][0][1]['is_visible']
        assert ctx['body'][1][2]['is_visible']
        ctx = df.style.hide('A', axis='columns')._translate(True, True)
        assert not ctx['head'][0][1]['is_visible']
        assert not ctx['body'][0][1]['is_visible']
        assert ctx['body'][1][2]['is_visible']
        ctx = df.style.hide(['A', 'B'], axis='columns')._translate(True, True)
        assert not ctx['head'][0][1]['is_visible']
        assert not ctx['head'][0][2]['is_visible']
        assert not ctx['body'][0][1]['is_visible']
        assert not ctx['body'][1][2]['is_visible']

    def test_hide_columns_index_mult_levels(self):
        i1 = MultiIndex.from_arrays([['a', 'a'], [0, 1]], names=['idx_level_0', 'idx_level_1'])
        i2 = MultiIndex.from_arrays([['b', 'b'], [0, 1]], names=['col_level_0', 'col_level_1'])
        df = DataFrame([[1, 2], [3, 4]], index=i1, columns=i2)
        ctx = df.style._translate(True, True)
        assert ctx['head'][0][2]['is_visible']
        assert ctx['head'][1][2]['is_visible']
        assert ctx['head'][1][3]['display_value'] == '1'
        assert ctx['body'][0][0]['is_visible']
        assert ctx['body'][1][2]['is_visible']
        assert ctx['body'][1][2]['display_value'] == '3'
        assert ctx['body'][1][3]['is_visible']
        assert ctx['body'][1][3]['display_value'] == '4'
        ctx = df.style.hide('b', axis='columns')._translate(True, True)
        assert not ctx['head'][0][2]['is_visible']
        assert not ctx['head'][1][2]['is_visible']
        assert not ctx['body'][1][2]['is_visible']
        assert ctx['body'][0][0]['is_visible']
        ctx = df.style.hide([('b', 0)], axis='columns')._translate(True, True)
        assert not ctx['head'][0][2]['is_visible']
        assert ctx['head'][0][3]['is_visible']
        assert not ctx['head'][1][2]['is_visible']
        assert not ctx['body'][1][2]['is_visible']
        assert ctx['body'][1][3]['is_visible']
        assert ctx['body'][1][3]['display_value'] == '4'
        ctx = df.style.hide([('b', 1)], axis=1).hide(axis=0)._translate(True, True)
        assert not ctx['body'][0][0]['is_visible']
        assert len(ctx['head'][0]) == 3
        assert ctx['head'][0][1]['is_visible']
        assert ctx['head'][1][1]['is_visible']
        assert not ctx['head'][1][2]['is_visible']
        assert not ctx['body'][1][3]['is_visible']
        assert ctx['body'][1][2]['is_visible']
        assert ctx['body'][1][2]['display_value'] == '3'
        ctx = df.style.hide('a', axis='index')._translate(True, True)
        assert ctx['body'] == []
        ctx = df.style.hide(('a', 0), axis='index')._translate(True, True)
        for i in [0, 1, 2, 3]:
            assert 'row1' in ctx['body'][0][i]['class']
            assert ctx['body'][0][i]['is_visible']

    def test_pipe(self, df):

        def set_caption_from_template(styler, a, b):
            return styler.set_caption(f'Dataframe with a = {a} and b = {b}')
        styler = df.style.pipe(set_caption_from_template, 'A', b='B')
        assert 'Dataframe with a = A and b = B' in styler.to_html()

        def f(a, b, styler):
            return (a, b, styler)
        styler = df.style
        result = styler.pipe((f, 'styler'), a=1, b=2)
        assert result == (1, 2, styler)

    def test_no_cell_ids(self):
        df = DataFrame(data=[[0]])
        styler = Styler(df, uuid='_', cell_ids=False)
        styler.to_html()
        s = styler.to_html()
        assert s.find('<td class="data row0 col0" >') != -1

    @pytest.mark.parametrize('classes', [DataFrame(data=[['', 'test-class'], [np.nan, None]], columns=['A', 'B'], index=['a', 'b']), DataFrame(data=[['test-class']], columns=['B'], index=['a']), DataFrame(data=[['test-class', 'unused']], columns=['B', 'C'], index=['a'])])
    def test_set_data_classes(self, classes):
        df = DataFrame(data=[[0, 1], [2, 3]], columns=['A', 'B'], index=['a', 'b'])
        s = Styler(df, uuid_len=0, cell_ids=False).set_td_classes(classes).to_html()
        assert '<td class="data row0 col0" >0</td>' in s
        assert '<td class="data row0 col1 test-class" >1</td>' in s
        assert '<td class="data row1 col0" >2</td>' in s
        assert '<td class="data row1 col1" >3</td>' in s
        s = Styler(df, uuid_len=0, cell_ids=True).set_td_classes(classes).to_html()
        assert '<td id="T__row0_col0" class="data row0 col0" >0</td>' in s
        assert '<td id="T__row0_col1" class="data row0 col1 test-class" >1</td>' in s
        assert '<td id="T__row1_col0" class="data row1 col0" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >3</td>' in s

    def test_set_data_classes_reindex(self):
        df = DataFrame(data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=[0, 1, 2], index=[0, 1, 2])
        classes = DataFrame(data=[['mi', 'ma'], ['mu', 'mo']], columns=[0, 2], index=[0, 2])
        s = Styler(df, uuid_len=0).set_td_classes(classes).to_html()
        assert '<td id="T__row0_col0" class="data row0 col0 mi" >0</td>' in s
        assert '<td id="T__row0_col2" class="data row0 col2 ma" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >4</td>' in s
        assert '<td id="T__row2_col0" class="data row2 col0 mu" >6</td>' in s
        assert '<td id="T__row2_col2" class="data row2 col2 mo" >8</td>' in s

    def test_chaining_table_styles(self):
        df = DataFrame(data=[[0, 1], [1, 2]], columns=['A', 'B'])
        styler = df.style.set_table_styles([{'selector': '', 'props': [('background-color', 'yellow')]}]).set_table_styles([{'selector': '.col0', 'props': [('background-color', 'blue')]}], overwrite=False)
        assert len(styler.table_styles) == 2

    def test_column_and_row_styling(self):
        df = DataFrame(data=[[0, 1], [1, 2]], columns=['A', 'B'])
        s = Styler(df, uuid_len=0)
        s = s.set_table_styles({'A': [{'selector': '', 'props': [('color', 'blue')]}]})
        assert '#T_ .col0 {\n  color: blue;\n}' in s.to_html()
        s = s.set_table_styles({0: [{'selector': '', 'props': [('color', 'blue')]}]}, axis=1)
        assert '#T_ .row0 {\n  color: blue;\n}' in s.to_html()

    @pytest.mark.parametrize('len_', [1, 5, 32, 33, 100])
    def test_uuid_len(self, len_):
        df = DataFrame(data=[['A']])
        s = Styler(df, uuid_len=len_, cell_ids=False).to_html()
        strt = s.find('id="T_')
        end = s[strt + 6:].find('"')
        if len_ > 32:
            assert end == 32
        else:
            assert end == len_

    @pytest.mark.parametrize('len_', [-2, 'bad', None])
    def test_uuid_len_raises(self, len_):
        df = DataFrame(data=[['A']])
        msg = '``uuid_len`` must be an integer in range \\[0, 32\\].'
        with pytest.raises(TypeError, match=msg):
            Styler(df, uuid_len=len_, cell_ids=False).to_html()

    @pytest.mark.parametrize('slc', [IndexSlice[:, :], IndexSlice[:, 1], IndexSlice[1, :], IndexSlice[[1], [1]], IndexSlice[1, [1]], IndexSlice[[1], 1], IndexSlice[1], IndexSlice[1, 1], slice(None, None, None), [0, 1], np.array([0, 1]), Series([0, 1])])
    def test_non_reducing_slice(self, slc):
        df = DataFrame([[0, 1], [2, 3]])
        tslice_ = non_reducing_slice(slc)
        assert isinstance(df.loc[tslice_], DataFrame)

    @pytest.mark.parametrize('box', [list, Series, np.array])
    def test_list_slice(self, box):
        subset = box(['A'])
        df = DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['A', 'B'])
        expected = IndexSlice[:, ['A']]
        result = non_reducing_slice(subset)
        tm.assert_frame_equal(df.loc[result], df.loc[expected])

    def test_non_reducing_slice_on_multiindex(self):
        dic = {('a', 'd'): [1, 4], ('a', 'c'): [2, 3], ('b', 'c'): [3, 2], ('b', 'd'): [4, 1]}
        df = DataFrame(dic, index=[0, 1])
        idx = IndexSlice
        slice_ = idx[:, idx['b', 'd']]
        tslice_ = non_reducing_slice(slice_)
        result = df.loc[tslice_]
        expected = DataFrame({('b', 'd'): [4, 1]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('slice_', [IndexSlice[:, :], IndexSlice[:, IndexSlice[['a']]], IndexSlice[:, IndexSlice[['a'], ['c']]], IndexSlice[:, IndexSlice['a', 'c', :]], IndexSlice[:, IndexSlice['a', :, 'e']], IndexSlice[:, IndexSlice[:, 'c', 'e']], IndexSlice[:, IndexSlice['a', ['c', 'd'], :]], IndexSlice[:, IndexSlice['a', ['c', 'd', '-'], :]], IndexSlice[:, IndexSlice['a', ['c', 'd', '-'], 'e']], IndexSlice[IndexSlice[['U']], :], IndexSlice[IndexSlice[['U'], ['W']], :], IndexSlice[IndexSlice['U', 'W', :], :], IndexSlice[IndexSlice['U', :, 'Y'], :], IndexSlice[IndexSlice[:, 'W', 'Y'], :], IndexSlice[IndexSlice[:, 'W', ['Y', 'Z']], :], IndexSlice[IndexSlice[:, 'W', ['Y', 'Z', '-']], :], IndexSlice[IndexSlice['U', 'W', ['Y', 'Z', '-']], :], IndexSlice[IndexSlice[:, 'W', 'Y'], IndexSlice['a', 'c', :]]])
    def test_non_reducing_multi_slice_on_multiindex(self, slice_):
        cols = MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']])
        idxs = MultiIndex.from_product([['U', 'V'], ['W', 'X'], ['Y', 'Z']])
        df = DataFrame(np.arange(64).reshape(8, 8), columns=cols, index=idxs)
        for lvl in [0, 1]:
            key = slice_[lvl]
            if isinstance(key, tuple):
                for subkey in key:
                    if isinstance(subkey, list) and '-' in subkey:
                        with pytest.raises(KeyError, match='-'):
                            df.loc[slice_]
                        return
        expected = df.loc[slice_]
        result = df.loc[non_reducing_slice(slice_)]
        tm.assert_frame_equal(result, expected)