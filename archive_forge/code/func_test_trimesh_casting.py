import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_trimesh_casting(self):
    trimesh = TriMesh(([(0, 1, 2)], [(0, 0, 0), (0, 1, 1), (1, 1, 2)]))
    self.assertEqual(trimesh, TriMesh(trimesh))