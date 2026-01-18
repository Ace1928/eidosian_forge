import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_single_path_list_constructor(self):
    path = Path([(0, 1), (1, 2), (2, 3), (3, 4)])
    self.assertEqual(path.dimension_values(0), np.array([0, 1, 2, 3]))
    self.assertEqual(path.dimension_values(1), np.array([1, 2, 3, 4]))