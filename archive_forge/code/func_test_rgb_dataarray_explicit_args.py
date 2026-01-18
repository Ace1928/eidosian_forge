import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_rgb_dataarray_explicit_args(self):
    rgb = self.da_rgb.hvplot('x', 'y')
    self.assertEqual(rgb, RGB(([0, 1], [0, 1]) + tuple(self.da_rgb.values)))