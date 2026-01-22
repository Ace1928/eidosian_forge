from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
class TestScalarFormatter:
    offset_data = [(123, 189, 0), (-189, -123, 0), (12341, 12349, 12340), (-12349, -12341, -12340), (99999.5, 100010.5, 100000), (-100010.5, -99999.5, -100000), (99990.5, 100000.5, 100000), (-100000.5, -99990.5, -100000), (1233999, 1234001, 1234000), (-1234001, -1233999, -1234000), (1, 1, 1), (123, 123, 0), (0.4538, 0.4578, 0.45), (3789.12, 3783.1, 3780), (45124.3, 45831.75, 45000), (0.000721, 0.0007243, 0.00072), (12592.82, 12591.43, 12590), (9.0, 12.0, 0), (900.0, 1200.0, 0), (1900.0, 1200.0, 0), (0.99, 1.01, 1), (9.99, 10.01, 10), (99.99, 100.01, 100), (5.99, 6.01, 6), (15.99, 16.01, 16), (-0.452, 0.492, 0), (-0.492, 0.492, 0), (12331.4, 12350.5, 12300), (-12335.3, 12335.3, 0)]
    use_offset_data = [True, False]
    useMathText_data = [True, False]
    scilimits_data = [(False, (0, 0), (10.0, 20.0), 0, False), (True, (-2, 2), (-10, 20), 0, False), (True, (-2, 2), (-20, 10), 0, False), (True, (-2, 2), (-110, 120), 2, False), (True, (-2, 2), (-120, 110), 2, False), (True, (-2, 2), (-0.001, 0.002), -3, False), (True, (-7, 7), (1800000000.0, 8300000000.0), 9, True), (True, (0, 0), (-100000.0, 100000.0), 5, False), (True, (6, 6), (-100000.0, 100000.0), 6, False)]
    cursor_data = [[0.0, '0.000'], [0.0123, '0.012'], [0.123, '0.123'], [1.23, '1.230'], [12.3, '12.300']]
    format_data = [(0.1, '1e-1'), (0.11, '1.1e-1'), (100000000.0, '1e8'), (110000000.0, '1.1e8')]

    @pytest.mark.parametrize('unicode_minus, result', [(True, '−1'), (False, '-1')])
    def test_unicode_minus(self, unicode_minus, result):
        mpl.rcParams['axes.unicode_minus'] = unicode_minus
        assert plt.gca().xaxis.get_major_formatter().format_data_short(-1).strip() == result

    @pytest.mark.parametrize('left, right, offset', offset_data)
    def test_offset_value(self, left, right, offset):
        fig, ax = plt.subplots()
        formatter = ax.xaxis.get_major_formatter()
        with pytest.warns(UserWarning, match='Attempting to set identical') if left == right else nullcontext():
            ax.set_xlim(left, right)
        ax.xaxis._update_ticks()
        assert formatter.offset == offset
        with pytest.warns(UserWarning, match='Attempting to set identical') if left == right else nullcontext():
            ax.set_xlim(right, left)
        ax.xaxis._update_ticks()
        assert formatter.offset == offset

    @pytest.mark.parametrize('use_offset', use_offset_data)
    def test_use_offset(self, use_offset):
        with mpl.rc_context({'axes.formatter.useoffset': use_offset}):
            tmp_form = mticker.ScalarFormatter()
            assert use_offset == tmp_form.get_useOffset()
            assert tmp_form.offset == 0

    @pytest.mark.parametrize('use_math_text', useMathText_data)
    def test_useMathText(self, use_math_text):
        with mpl.rc_context({'axes.formatter.use_mathtext': use_math_text}):
            tmp_form = mticker.ScalarFormatter()
            assert use_math_text == tmp_form.get_useMathText()

    def test_set_use_offset_float(self):
        tmp_form = mticker.ScalarFormatter()
        tmp_form.set_useOffset(0.5)
        assert not tmp_form.get_useOffset()
        assert tmp_form.offset == 0.5

    def test_use_locale(self):
        conv = locale.localeconv()
        sep = conv['thousands_sep']
        if not sep or conv['grouping'][-1:] in ([], [locale.CHAR_MAX]):
            pytest.skip('Locale does not apply grouping')
        with mpl.rc_context({'axes.formatter.use_locale': True}):
            tmp_form = mticker.ScalarFormatter()
            assert tmp_form.get_useLocale()
            tmp_form.create_dummy_axis()
            tmp_form.axis.set_data_interval(0, 10)
            tmp_form.set_locs([1, 2, 3])
            assert sep in tmp_form(1000000000.0)

    @pytest.mark.parametrize('sci_type, scilimits, lim, orderOfMag, fewticks', scilimits_data)
    def test_scilimits(self, sci_type, scilimits, lim, orderOfMag, fewticks):
        tmp_form = mticker.ScalarFormatter()
        tmp_form.set_scientific(sci_type)
        tmp_form.set_powerlimits(scilimits)
        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(tmp_form)
        ax.set_ylim(*lim)
        if fewticks:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        tmp_form.set_locs(ax.yaxis.get_majorticklocs())
        assert orderOfMag == tmp_form.orderOfMagnitude

    @pytest.mark.parametrize('value, expected', format_data)
    def test_format_data(self, value, expected):
        mpl.rcParams['axes.unicode_minus'] = False
        sf = mticker.ScalarFormatter()
        assert sf.format_data(value) == expected

    @pytest.mark.parametrize('data, expected', cursor_data)
    def test_cursor_precision(self, data, expected):
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        fmt = ax.xaxis.get_major_formatter().format_data_short
        assert fmt(data) == expected

    @pytest.mark.parametrize('data, expected', cursor_data)
    def test_cursor_dummy_axis(self, data, expected):
        sf = mticker.ScalarFormatter()
        sf.create_dummy_axis()
        sf.axis.set_view_interval(0, 10)
        fmt = sf.format_data_short
        assert fmt(data) == expected
        assert sf.axis.get_tick_space() == 9
        assert sf.axis.get_minpos() == 0

    def test_mathtext_ticks(self):
        mpl.rcParams.update({'font.family': 'serif', 'font.serif': 'cmr10', 'axes.formatter.use_mathtext': False})
        if parse_version(pytest.__version__).major < 8:
            with pytest.warns(UserWarning, match='cmr10 font should ideally'):
                fig, ax = plt.subplots()
                ax.set_xticks([-1, 0, 1])
                fig.canvas.draw()
        else:
            with pytest.warns(UserWarning, match='Glyph 8722'), pytest.warns(UserWarning, match='cmr10 font should ideally'):
                fig, ax = plt.subplots()
                ax.set_xticks([-1, 0, 1])
                fig.canvas.draw()

    def test_cmr10_substitutions(self, caplog):
        mpl.rcParams.update({'font.family': 'cmr10', 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True})
        with caplog.at_level(logging.WARNING, logger='matplotlib.mathtext'):
            fig, ax = plt.subplots()
            ax.plot([-0.03, 0.05], [40, 0.05])
            ax.set_yscale('log')
            yticks = [0.02, 0.3, 4, 50]
            formatter = mticker.LogFormatterSciNotation()
            ax.set_yticks(yticks, map(formatter, yticks))
            fig.canvas.draw()
            assert not caplog.text

    def test_empty_locs(self):
        sf = mticker.ScalarFormatter()
        sf.set_locs([])
        assert sf(0.5) == ''