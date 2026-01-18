from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_slice_exclusive_upper(self):
    """Exclusive upper boundary semantics for bin centers"""
    sliced = self.dataset1d[:6.5]
    self.assertEqual(sliced.dimension_values(1), np.arange(6))
    self.assertEqual(sliced.edges, np.arange(7))