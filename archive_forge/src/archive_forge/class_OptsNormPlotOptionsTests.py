from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
class OptsNormPlotOptionsTests(ComparisonTestCase):
    """
    Test the OptsSpec parser works correctly for plot options.
    """

    def test_norm_opts_simple_1(self):
        line = 'Layout {+axiswise}'
        expected = {'Layout': {'norm': Options(axiswise=True, framewise=False)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_explicit_1(self):
        line = 'Layout norm{+axiswise}'
        expected = {'Layout': {'norm': Options(axiswise=True, framewise=False)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_2(self):
        line = 'Layout {+axiswise +framewise}'
        expected = {'Layout': {'norm': Options(axiswise=True, framewise=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_norm_opts_simple_explicit_2(self):
        line = 'Layout norm{+axiswise +framewise}'
        expected = {'Layout': {'norm': Options(axiswise=True, framewise=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_norm_opts_multiple_paths(self):
        line = 'Image Curve {+axiswise +framewise}'
        expected = {'Image': {'norm': Options(axiswise=True, framewise=True)}, 'Curve': {'norm': Options(axiswise=True, framewise=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)