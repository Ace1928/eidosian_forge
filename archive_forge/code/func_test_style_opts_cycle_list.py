from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_style_opts_cycle_list(self):
    line = "Curve (color=Cycle(values=['r', 'g', 'b']))"
    expected = {'Curve': {'style': Options(color=Cycle(values=['r', 'g', 'b']))}}
    self.assertEqual(OptsSpec.parse(line, {'Cycle': Cycle}), expected)