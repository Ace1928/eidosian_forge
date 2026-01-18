import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_spec_duplicate_dim_select(self):
    selection = self.duplicate_map.select(selection_specs=(HoloMap,), x=(0, 1), y=(1, 3))
    self.assertEqual(selection, self.duplicate_map[0:1, 1:3])