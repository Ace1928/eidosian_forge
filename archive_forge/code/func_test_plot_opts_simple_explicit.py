from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_plot_opts_simple_explicit(self):
    line = "Layout plot[fig_inches=(3,3) title='foo bar']"
    expected = {'Layout': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
    self.assertEqual(OptsSpec.parse(line), expected)