from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_plot_opts_dict_without_space(self):
    line = 'Curve [fontsize=dict(xlabel=10,title=20)]'
    expected = {'Curve': {'plot': Options(fontsize={'xlabel': 10, 'title': 20})}}
    self.assertEqual(OptsSpec.parse(line), expected)