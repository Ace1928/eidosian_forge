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
def test_uneven_edges(self):
    self.assertEqual(compute_edges(self.array3), np.array([0.5, 1.5, 3.0, 5.0]))