import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
def test_rasterize_trimesh_vertex_vdims(self):
    simplices = [(0, 1, 2), (3, 2, 1)]
    vertices = [(0.0, 0.0, 1), (0.0, 1.0, 2), (1.0, 0.0, 3), (1.0, 1.0, 4)]
    trimesh = TriMesh((simplices, Points(vertices, vdims='z')))
    img = rasterize(trimesh, width=3, height=3, dynamic=False)
    array = np.array([[2.166667, 2.833333, 3.5], [1.833333, 2.5, 3.166667], [1.5, 2.166667, 2.833333]])
    image = Image(array, bounds=(0, 0, 1, 1), vdims='z')
    self.assertEqual(img, image)