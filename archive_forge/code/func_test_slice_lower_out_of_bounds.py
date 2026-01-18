from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_slice_lower_out_of_bounds(self):
    sliced = self.dataset1d[-3:]
    self.assertEqual(sliced.dimension_values(1), self.values)
    self.assertEqual(sliced.edges, self.edges)