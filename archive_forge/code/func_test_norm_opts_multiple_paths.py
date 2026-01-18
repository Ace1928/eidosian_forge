from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_norm_opts_multiple_paths(self):
    line = 'Image Curve {+axiswise +framewise}'
    expected = {'Image': {'norm': Options(axiswise=True, framewise=True)}, 'Curve': {'norm': Options(axiswise=True, framewise=True)}}
    self.assertEqual(OptsSpec.parse(line), expected)