import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_single_path_split(self):
    path = Path(([0, 1, 2, 3], [1, 2, 3, 4]))
    self.assertEqual(path, path.split()[0])