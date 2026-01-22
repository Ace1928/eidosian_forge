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
class DatashaderRasterizeTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def setUp(self):
        if ds_version <= Version('0.6.4'):
            raise SkipTest('Regridding operations require datashader>=0.7.0')
        self.simplexes = [(0, 1, 2), (3, 2, 1)]
        self.vertices = [(0.0, 0.0), (0.0, 1.0), (1.0, 0), (1, 1)]
        self.simplexes_vdim = [(0, 1, 2, 0.5), (3, 2, 1, 1.5)]
        self.vertices_vdim = [(0.0, 0.0, 1), (0.0, 1.0, 2), (1.0, 0, 3), (1, 1, 4)]

    def test_rasterize_trimesh_no_vdims(self):
        trimesh = TriMesh((self.simplexes, self.vertices))
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        image = Image(np.array([[True, True, True], [True, True, True], [True, True, True]]), bounds=(0, 0, 1, 1), vdims=Dimension('Any', nodata=0))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_no_vdims_zero_range(self):
        trimesh = TriMesh((self.simplexes, self.vertices))
        img = rasterize(trimesh, height=2, x_range=(0, 0), dynamic=False)
        image = Image(([], [0.25, 0.75], np.zeros((2, 0))), bounds=(0, 0, 0, 1), xdensity=1, vdims=Dimension('Any', nodata=0))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_with_vdims_as_wireframe(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, aggregator='any', interpolation=None, dynamic=False)
        array = np.array([[True, True, True], [True, True, True], [True, True, True]])
        image = Image(array, bounds=(0, 0, 1, 1), vdims=Dimension('Any', nodata=0))
        self.assertEqual(img, image)

    def test_rasterize_trimesh(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        array = np.array([[0.5, 1.5, 1.5], [0.5, 0.5, 1.5], [0.5, 0.5, 0.5]])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_pandas_trimesh_implicit_nodes(self):
        simplex_df = pd.DataFrame(self.simplexes, columns=['v0', 'v1', 'v2'])
        vertex_df = pd.DataFrame(self.vertices_vdim, columns=['x', 'y', 'z'])
        trimesh = TriMesh((simplex_df, vertex_df))
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        array = np.array([[2.166667, 2.833333, 3.5], [1.833333, 2.5, 3.166667], [1.5, 2.166667, 2.833333]])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_dask_trimesh_implicit_nodes(self):
        simplex_df = pd.DataFrame(self.simplexes, columns=['v0', 'v1', 'v2'])
        vertex_df = pd.DataFrame(self.vertices_vdim, columns=['x', 'y', 'z'])
        simplex_ddf = dd.from_pandas(simplex_df, npartitions=2)
        vertex_ddf = dd.from_pandas(vertex_df, npartitions=2)
        trimesh = TriMesh((simplex_ddf, vertex_ddf))
        ri = rasterize.instance()
        img = ri(trimesh, width=3, height=3, dynamic=False, precompute=True)
        cache = ri._precomputed
        self.assertEqual(len(cache), 1)
        self.assertIn(trimesh._plot_id, cache)
        self.assertIsInstance(cache[trimesh._plot_id]['mesh'], dd.DataFrame)
        array = np.array([[2.166667, 2.833333, 3.5], [1.833333, 2.5, 3.166667], [1.5, 2.166667, 2.833333]])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

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

    def test_rasterize_dask_trimesh_with_node_vdims(self):
        simplex_df = pd.DataFrame(self.simplexes, columns=['v0', 'v1', 'v2'])
        vertex_df = pd.DataFrame(self.vertices_vdim, columns=['x', 'y', 'z'])
        simplex_ddf = dd.from_pandas(simplex_df, npartitions=2)
        vertex_ddf = dd.from_pandas(vertex_df, npartitions=2)
        tri_nodes = Nodes(vertex_ddf, ['x', 'y', 'index'], ['z'])
        trimesh = TriMesh((simplex_ddf, tri_nodes))
        ri = rasterize.instance()
        img = ri(trimesh, width=3, height=3, dynamic=False, precompute=True)
        cache = ri._precomputed
        self.assertEqual(len(cache), 1)
        self.assertIn(trimesh._plot_id, cache)
        self.assertIsInstance(cache[trimesh._plot_id]['mesh'], dd.DataFrame)
        array = np.array([[2.166667, 2.833333, 3.5], [1.833333, 2.5, 3.166667], [1.5, 2.166667, 2.833333]])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_node_vdim_precedence(self):
        nodes = Points(self.vertices_vdim, vdims=['node_z'])
        trimesh = TriMesh((self.simplexes_vdim, nodes), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        array = np.array([[2.166667, 2.833333, 3.5], [1.833333, 2.5, 3.166667], [1.5, 2.166667, 2.833333]])
        image = Image(array, bounds=(0, 0, 1, 1), vdims='node_z')
        self.assertEqual(img, image)

    def test_rasterize_trimesh_node_explicit_vdim(self):
        nodes = Points(self.vertices_vdim, vdims=['node_z'])
        trimesh = TriMesh((self.simplexes_vdim, nodes), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
        array = np.array([[0.5, 1.5, 1.5], [0.5, 0.5, 1.5], [0.5, 0.5, 0.5]])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_zero_range(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, x_range=(0, 0), height=2, dynamic=False)
        image = Image(([], [0.25, 0.75], np.zeros((2, 0))), bounds=(0, 0, 0, 1), xdensity=1)
        self.assertEqual(img, image)

    def test_rasterize_trimesh_vertex_vdims(self):
        simplices = [(0, 1, 2), (3, 2, 1)]
        vertices = [(0.0, 0.0, 1), (0.0, 1.0, 2), (1.0, 0.0, 3), (1.0, 1.0, 4)]
        trimesh = TriMesh((simplices, Points(vertices, vdims='z')))
        img = rasterize(trimesh, width=3, height=3, dynamic=False)
        array = np.array([[2.166667, 2.833333, 3.5], [1.833333, 2.5, 3.166667], [1.5, 2.166667, 2.833333]])
        image = Image(array, bounds=(0, 0, 1, 1), vdims='z')
        self.assertEqual(img, image)

    def test_rasterize_trimesh_ds_aggregator(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
        array = np.array([[0.5, 1.5, 1.5], [0.5, 0.5, 1.5], [0.5, 0.5, 0.5]])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_trimesh_string_aggregator(self):
        trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
        img = rasterize(trimesh, width=3, height=3, dynamic=False, aggregator='mean')
        array = np.array([[0.5, 1.5, 1.5], [0.5, 0.5, 1.5], [0.5, 0.5, 0.5]])
        image = Image(array, bounds=(0, 0, 1, 1))
        self.assertEqual(img, image)

    def test_rasterize_quadmesh(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
        img = rasterize(qmesh, width=3, height=3, dynamic=False, aggregator=ds.mean('z'))
        image = Image(np.array([[2, 3, 3], [2, 3, 3], [0, 1, 1]]), bounds=(-0.5, -0.5, 1.5, 1.5))
        self.assertEqual(img, image)

    def test_rasterize_quadmesh_string_aggregator(self):
        qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [2, 3]])))
        img = rasterize(qmesh, width=3, height=3, dynamic=False, aggregator='mean')
        image = Image(np.array([[2, 3, 3], [2, 3, 3], [0, 1, 1]]), bounds=(-0.5, -0.5, 1.5, 1.5))
        self.assertEqual(img, image)

    def test_rasterize_points(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        img = rasterize(points, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2)
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]), vdims=[Dimension('Count', nodata=0)])
        self.assertEqual(img, expected)

    def test_rasterize_curve(self):
        curve = Curve([(0.2, 0.3), (0.4, 0.7), (0.8, 0.99)])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [1, 1]]), vdims=[Dimension('Count', nodata=0)])
        img = rasterize(curve, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2)
        self.assertEqual(img, expected)

    def test_rasterize_ndoverlay(self):
        ds = Dataset([(0.2, 0.3, 0), (0.4, 0.7, 1), (0, 0.99, 2)], kdims=['x', 'y', 'z'])
        ndoverlay = ds.to(Points, ['x', 'y'], [], 'z').overlay()
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]), vdims=[Dimension('Count', nodata=0)])
        img = rasterize(ndoverlay, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2)
        self.assertEqual(img, expected)

    def test_rasterize_path(self):
        path = Path([[(0.2, 0.3), (0.4, 0.7)], [(0.4, 0.7), (0.8, 0.99)]])
        expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 1]]), vdims=[Dimension('Count', nodata=0)])
        img = rasterize(path, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2)
        self.assertEqual(img, expected)

    def test_rasterize_image(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, width=2, height=2, dynamic=False)
        expected = Image(([2.0, 7.0], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_rasterize_image_string_aggregator(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, width=2, height=2, dynamic=False, aggregator='mean')
        expected = Image(([2.0, 7.0], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_rasterize_image_expand_default(self):
        assert not regrid.expand
        data = np.arange(100.0).reshape(10, 10)
        c = np.arange(10.0)
        da = xr.DataArray(data, coords=dict(x=c, y=c))
        rast_input = dict(x_range=(-1, 10), y_range=(-1, 10), precompute=True, dynamic=False)
        img = rasterize(Image(da), **rast_input)
        output = img.data['z'].to_numpy()
        np.testing.assert_array_equal(output, data.T)
        assert not np.isnan(output).any()
        img = rasterize(Image(da), expand=True, **rast_input)
        output = img.data['z'].to_numpy()
        assert np.isnan(output).any()

    def test_rasterize_apply_when_instance_with_line_width(self):
        df = pd.DataFrame(np.random.multivariate_normal((0, 0), [[0.1, 0.1], [0.1, 1.0]], (100,)))
        df.columns = ['a', 'b']
        curve = Curve(df, kdims=['a'], vdims=['b'])
        custom_rasterize = rasterize.instance(line_width=2)
        assert {'line_width': 2} == custom_rasterize._rasterize__instance_kwargs
        output = apply_when(curve, operation=custom_rasterize, predicate=lambda x: len(x) > 10)
        render(output, 'bokeh')
        assert isinstance(output, DynamicMap)
        overlay = output.items()[0][1]
        assert isinstance(overlay, Overlay)
        assert len(overlay) == 2