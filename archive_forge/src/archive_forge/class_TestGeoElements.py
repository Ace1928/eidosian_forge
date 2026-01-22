import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
class TestGeoElements(TestCase):

    def setUp(self):
        try:
            import geoviews
            import cartopy.crs as ccrs
        except:
            raise SkipTest('geoviews or cartopy not available')
        import hvplot.pandas
        self.crs = ccrs.PlateCarree()
        self.df = pd.DataFrame(np.random.rand(10, 2), columns=['x', 'y'])

    def test_geo_hexbin(self):
        hextiles = self.df.hvplot.hexbin('x', 'y', geo=True)
        self.assertEqual(hextiles.crs, self.crs)

    def test_geo_points(self):
        points = self.df.hvplot.points('x', 'y', geo=True)
        self.assertEqual(points.crs, self.crs)

    def test_geo_points_color_internally_set_to_dim(self):
        altered_df = self.df.copy().assign(red=np.random.choice(['a', 'b'], len(self.df)))
        plot = altered_df.hvplot.points('x', 'y', c='red', geo=True)
        opts = hv.Store.lookup_options('bokeh', plot, 'style')
        self.assertIsInstance(opts.kwargs['color'], hv.dim)
        self.assertEqual(opts.kwargs['color'].dimension.name, 'red')

    def test_geo_opts(self):
        points = self.df.hvplot.points('x', 'y', geo=True)
        opts = hv.Store.lookup_options('bokeh', points, 'plot').kwargs
        self.assertEqual(opts.get('data_aspect'), 1)
        self.assertEqual(opts.get('width'), None)

    def test_geo_opts_with_width(self):
        points = self.df.hvplot.points('x', 'y', geo=True, width=200)
        opts = hv.Store.lookup_options('bokeh', points, 'plot').kwargs
        self.assertEqual(opts.get('data_aspect'), 1)
        self.assertEqual(opts.get('width'), 200)
        self.assertEqual(opts.get('height'), None)