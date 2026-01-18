import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_rgb_dataarray_groupby_explicit(self):
    rgb = self.da_rgb_by_time.hvplot.rgb('x', 'y', groupby='time')
    self.assertEqual(rgb[0], RGB(([0, 1], [0, 1]) + tuple(self.da_rgb_by_time.values[0])))
    self.assertEqual(rgb[1], RGB(([0, 1], [0, 1]) + tuple(self.da_rgb_by_time.values[1])))