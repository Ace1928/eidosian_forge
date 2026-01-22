import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
class TestProjections(TestGeo):

    def test_plot_with_crs_as_object(self):
        plot = self.da.hvplot.image('x', 'y', crs=self.crs)
        self.assertCRS(plot)

    def test_plot_with_crs_as_attr_str(self):
        da = self.da.copy()
        da.rio._crs = False
        da.attrs = {'bar': self.crs}
        plot = da.hvplot.image('x', 'y', crs='bar')
        self.assertCRS(plot)

    def test_plot_with_crs_as_nonexistent_attr_str(self):
        da = self.da.copy()
        da.rio._crs = False
        with self.assertRaisesRegex(ValueError, "'name_of_some_invalid_projection' must be"):
            da.hvplot.image('x', 'y', crs='name_of_some_invalid_projection')

    def test_plot_with_geo_as_true_crs_no_crs_on_data_returns_default(self):
        da = self.da.copy()
        da.rio._crs = False
        da.attrs = {'bar': self.crs}
        plot = da.hvplot.image('x', 'y', geo=True)
        self.assertCRS(plot, 'eqc')

    def test_plot_with_projection_as_string(self):
        da = self.da.copy()
        plot = da.hvplot.image('x', 'y', crs=self.crs, projection='Robinson')
        self.assert_projection(plot, 'robin')

    def test_plot_with_projection_as_string_google_mercator(self):
        da = self.da.copy()
        plot = da.hvplot.image('x', 'y', crs=self.crs, projection='GOOGLE_MERCATOR')
        self.assert_projection(plot, 'merc')

    def test_plot_with_projection_as_invalid_string(self):
        with self.assertRaisesRegex(ValueError, 'Projection must be defined'):
            self.da.hvplot.image('x', 'y', projection='foo')

    def test_plot_with_projection_raises_an_error_when_tiles_set(self):
        da = self.da.copy()
        with self.assertRaisesRegex(ValueError, 'Tiles can only be used with output projection'):
            da.hvplot.image('x', 'y', crs=self.crs, projection='Robinson', tiles=True)

    def test_overlay_with_projection(self):
        df = pd.DataFrame({'lon': [0, 10], 'lat': [40, 50], 'v': [0, 1]})
        plot1 = df.hvplot.points(x='lon', y='lat', s=200, c='y', geo=True, tiles='CartoLight')
        plot2 = df.hvplot.points(x='lon', y='lat', c='v', geo=True)
        plot = plot1 * plot2
        hv.renderer('bokeh').get_plot(plot)

    def test_geo_with_rasterize(self):
        import xarray as xr
        import cartopy.crs as ccrs
        import geoviews as gv
        try:
            from holoviews.operation.datashader import rasterize
        except:
            raise SkipTest('datashader not available')
        ds = xr.tutorial.open_dataset('air_temperature')
        hvplot_output = ds.isel(time=0).hvplot.points('lon', 'lat', crs=ccrs.PlateCarree(), projection=ccrs.LambertConformal(), rasterize=True, dynamic=False, aggregator='max', project=True)
        p1 = gv.Points(ds.isel(time=0), kdims=['lon', 'lat'], crs=ccrs.PlateCarree())
        p2 = gv.project(p1, projection=ccrs.LambertConformal())
        expected = rasterize(p2, dynamic=False, aggregator='max')
        xr.testing.assert_allclose(hvplot_output.data, expected.data)