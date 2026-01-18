from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_style_opts_intermediate(self):
    line = "Layout (string='foo' test=3, b=True)"
    expected = {'Layout': {'style': Options(string='foo', test=3, b=True)}}
    self.assertEqual(OptsSpec.parse(line), expected)