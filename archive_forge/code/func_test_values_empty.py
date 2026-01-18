import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_values_empty(self):
    mds = Path([], kdims=['x', 'y'], datatype=[self.datatype])
    self.assertIs(mds.interface, self.interface)
    self.assertEqual(mds.dimension_values(0), np.array([]))