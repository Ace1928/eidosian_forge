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
def test_rasterize_trimesh_node_explicit_vdim(self):
    nodes = Points(self.vertices_vdim, vdims=['node_z'])
    trimesh = TriMesh((self.simplexes_vdim, nodes), vdims=['z'])
    img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
    array = np.array([[0.5, 1.5, 1.5], [0.5, 0.5, 1.5], [0.5, 0.5, 0.5]])
    image = Image(array, bounds=(0, 0, 1, 1))
    self.assertEqual(img, image)