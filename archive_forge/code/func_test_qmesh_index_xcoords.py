from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_qmesh_index_xcoords(self):
    sliced = QuadMesh((self.xs[2:4], self.ys, self.zs[:, 2:3]))
    self.assertEqual(self.dataset2d[300, :], sliced)