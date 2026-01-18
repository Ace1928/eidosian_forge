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
def test_rasterize_dask_trimesh(self):
    simplex_df = pd.DataFrame(self.simplexes_vdim, columns=['v0', 'v1', 'v2', 'z'])
    vertex_df = pd.DataFrame(self.vertices, columns=['x', 'y'])
    simplex_ddf = dd.from_pandas(simplex_df, npartitions=2)
    vertex_ddf = dd.from_pandas(vertex_df, npartitions=2)
    tri_nodes = Nodes(vertex_ddf, ['x', 'y', 'index'])
    trimesh = TriMesh((simplex_ddf, tri_nodes), vdims=['z'])
    ri = rasterize.instance()
    img = ri(trimesh, width=3, height=3, dynamic=False, precompute=True)
    cache = ri._precomputed
    self.assertEqual(len(cache), 1)
    self.assertIn(trimesh._plot_id, cache)
    self.assertIsInstance(cache[trimesh._plot_id]['mesh'], dd.DataFrame)
    array = np.array([[0.5, 1.5, 1.5], [0.5, 0.5, 1.5], [0.5, 0.5, 0.5]])
    image = Image(array, bounds=(0, 0, 1, 1))
    self.assertEqual(img, image)