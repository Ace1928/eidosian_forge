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
def test_rasterize_trimesh_with_vdims_as_wireframe(self):
    trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
    img = rasterize(trimesh, width=3, height=3, aggregator='any', interpolation=None, dynamic=False)
    array = np.array([[True, True, True], [True, True, True], [True, True, True]])
    image = Image(array, bounds=(0, 0, 1, 1), vdims=Dimension('Any', nodata=0))
    self.assertEqual(img, image)