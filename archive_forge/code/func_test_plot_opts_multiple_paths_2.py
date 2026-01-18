from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_plot_opts_multiple_paths_2(self):
    line = "Image Curve Layout Overlay[fig_inches=(3, 3) title='foo bar']"
    expected = {'Image': {'plot': Options(title='foo bar', fig_inches=(3, 3))}, 'Curve': {'plot': Options(title='foo bar', fig_inches=(3, 3))}, 'Layout': {'plot': Options(title='foo bar', fig_inches=(3, 3))}, 'Overlay': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
    self.assertEqual(OptsSpec.parse(line), expected)