import numpy as np
import param
import pytest
from packaging.version import Version
from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
def test_vline_dimension_values(self):
    vline = VLine(0)
    self.assertEqual(vline.range(0), (0, 0))
    self.assertTrue(all((not np.isfinite(v) for v in vline.range(1))))
    vline = VLine(np.array([0]))
    self.assertEqual(vline.range(0), (0, 0))
    self.assertTrue(all((not np.isfinite(v) for v in vline.range(1))))
    vline = VLine(np.array(0))
    self.assertEqual(vline.range(0), (0, 0))
    self.assertTrue(all((not np.isfinite(v) for v in vline.range(1))))