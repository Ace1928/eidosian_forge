import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_overlap_select(self):
    selection = self.overlap_layout.select(Default=(6, None))
    self.assertEqual(selection, self.overlap1.clone(shared_data=False) + self.overlap2[6:])