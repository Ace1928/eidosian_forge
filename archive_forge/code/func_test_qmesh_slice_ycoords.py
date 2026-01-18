from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_qmesh_slice_ycoords(self):
    sliced = QuadMesh((self.xs, self.ys[:-1], self.zs[:-1, :]))
    self.assertEqual(self.dataset2d[:, 2:7], sliced)