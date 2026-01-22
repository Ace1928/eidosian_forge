import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
class TestBoxWithBy:

    @pytest.mark.parametrize('by, column, titles, xticklabels', [('C', 'A', ['A'], [['a', 'b', 'c']]), (['C', 'D'], 'A', ['A'], [['(a, a)', '(b, b)', '(c, c)']]), ('C', ['A', 'B'], ['A', 'B'], [['a', 'b', 'c']] * 2), (['C', 'D'], ['A', 'B'], ['A', 'B'], [['(a, a)', '(b, b)', '(c, c)']] * 2), (['C'], None, ['A', 'B'], [['a', 'b', 'c']] * 2)])
    def test_box_plot_by_argument(self, by, column, titles, xticklabels, hist_df):
        axes = _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_xticklabels = [[label.get_text() for label in ax.get_xticklabels()] for ax in axes]
        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize('by, column, titles, xticklabels', [(0, 'A', ['A'], [['a', 'b', 'c']]), ([0, 'D'], 'A', ['A'], [['(a, a)', '(b, b)', '(c, c)']]), (0, None, ['A', 'B'], [['a', 'b', 'c']] * 2)])
    def test_box_plot_by_0(self, by, column, titles, xticklabels, hist_df):
        df = hist_df.copy()
        df = df.rename(columns={'C': 0})
        axes = _check_plot_works(df.plot.box, default_axes=True, column=column, by=by)
        result_titles = [ax.get_title() for ax in axes]
        result_xticklabels = [[label.get_text() for label in ax.get_xticklabels()] for ax in axes]
        assert result_xticklabels == xticklabels
        assert result_titles == titles

    @pytest.mark.parametrize('by, column', [([], ['A']), ((), 'A'), ([], None), ((), ['A', 'B'])])
    def test_box_plot_with_none_empty_list_by(self, by, column, hist_df):
        msg = 'No group keys passed'
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)

    @pytest.mark.slow
    @pytest.mark.parametrize('by, column, layout, axes_num', [(['C'], 'A', (1, 1), 1), ('C', 'A', (1, 1), 1), ('C', None, (2, 1), 2), ('C', ['A', 'B'], (1, 2), 2), (['C', 'D'], 'A', (1, 1), 1), (['C', 'D'], None, (1, 2), 2)])
    def test_box_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
        axes = _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by, layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    @pytest.mark.parametrize('msg, by, layout', [('larger than required size', ['C', 'D'], (1, 1)), (re.escape('Layout must be a tuple of (rows, columns)'), 'C', (1,)), ('At least one dimension of layout must be positive', 'C', (-1, -1))])
    def test_box_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df):
        with pytest.raises(ValueError, match=msg):
            hist_df.plot.box(column=['A', 'B'], by=by, layout=layout)

    @pytest.mark.parametrize('figsize', [(12, 8), (20, 10)])
    def test_figure_shape_hist_with_by(self, figsize, hist_df):
        axes = hist_df.plot.box(column='A', by='C', figsize=figsize)
        _check_axes_shape(axes, axes_num=1, figsize=figsize)