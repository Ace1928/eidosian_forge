import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_simple_holo_ndoverlay_slice(self):
    self.assertEqual(self.ndoverlay_map.select(a=(1, 3), b=(1, 3)), self.ndoverlay_map[1:3, 1:3])