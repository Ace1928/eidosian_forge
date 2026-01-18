import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_qmesh_contours(self):
    qmesh = QuadMesh(([0, 1, 2], [1, 2, 3], np.array([[0, 1, 0], [3, 4, 5.0], [6, 7, 8]])))
    op_contours = contours(qmesh, levels=[0.5])
    contour = Contours([[(0, 1.166667, 0.5), (0.5, 1.0, 0.5), (np.nan, np.nan, 0.5), (1.5, 1.0, 0.5), (2, 1.1, 0.5)]], vdims=qmesh.vdims)
    self.assertEqual(op_contours, contour)