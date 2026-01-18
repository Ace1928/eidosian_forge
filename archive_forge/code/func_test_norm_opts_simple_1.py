from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_norm_opts_simple_1(self):
    line = 'Layout {+axiswise}'
    expected = {'Layout': {'norm': Options(axiswise=True, framewise=False)}}
    self.assertEqual(OptsSpec.parse(line), expected)