import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_sanitized_holoslice(self):
    self.assertEqual(self.sanitized_map.select(A_B=(1, 3)), self.sanitized_map[1:3])