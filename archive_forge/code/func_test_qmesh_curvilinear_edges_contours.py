import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_qmesh_curvilinear_edges_contours(self):
    x = y = np.arange(3)
    xs, ys = np.meshgrid(x, y)
    xs = GridInterface._infer_interval_breaks(xs)
    xs = GridInterface._infer_interval_breaks(xs, 1)
    ys = GridInterface._infer_interval_breaks(ys)
    ys = GridInterface._infer_interval_breaks(ys, 1)
    zs = np.array([[0, 1, 0], [3, 4, 5.0], [6, 7, 8]])
    qmesh = QuadMesh((xs, ys + 0.1, zs))
    op_contours = contours(qmesh, levels=[0.5])
    contour = Contours([[(0, 0.266667, 0.5), (0.5, 0.1, 0.5), (np.nan, np.nan, 0.5), (1.5, 0.1, 0.5), (2, 0.2, 0.5)]], vdims=qmesh.vdims)
    self.assertEqual(op_contours, contour)