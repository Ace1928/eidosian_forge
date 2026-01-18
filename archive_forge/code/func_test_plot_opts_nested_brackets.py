from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_plot_opts_nested_brackets(self):
    line = "Curve [title=', '.join(('A', 'B'))]"
    expected = {'Curve': {'plot': Options(title='A, B')}}
    self.assertEqual(OptsSpec.parse(line), expected)