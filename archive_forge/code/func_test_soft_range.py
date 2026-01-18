import datetime
import math
import unittest
from itertools import product
import numpy as np
import pandas as pd
from holoviews import Dimension, Element
from holoviews.core.util import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY
def test_soft_range(self):
    self.assertEqual(find_range(self.float_vals, soft_range=(np.nan, 100)), (-0.1424, 100))