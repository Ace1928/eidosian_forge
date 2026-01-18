from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_style_opts_simple_explicit(self):
    line = "Layout style(string='foo')"
    expected = {'Layout': {'style': Options(string='foo')}}
    self.assertEqual(OptsSpec.parse(line), expected)