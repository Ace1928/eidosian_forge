from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_style_opts_multiple_paths(self):
    line = "Image Curve (color='beautiful')"
    expected = {'Image': {'style': Options(color='beautiful')}, 'Curve': {'style': Options(color='beautiful')}}
    self.assertEqual(OptsSpec.parse(line), expected)