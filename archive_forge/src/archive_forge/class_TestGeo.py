import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
class TestGeo(TestCase):

    def setUp(self):
        if sys.platform == 'win32':
            raise SkipTest('Skip geo tests on windows for now')
        try:
            import xarray as xr
            import rasterio
            import geoviews
            import cartopy.crs as ccrs
            import rioxarray as rxr
        except:
            raise SkipTest('xarray, rasterio, geoviews, cartopy, or rioxarray not available')
        import hvplot.xarray
        import hvplot.pandas
        self.da = rxr.open_rasterio(pathlib.Path(__file__).parent / 'data' / 'RGB-red.byte.tif').isel(band=0)
        self.crs = proj_to_cartopy(self.da.spatial_ref.attrs['crs_wkt'])

    def assertCRS(self, plot, proj='utm'):
        import cartopy
        if Version(cartopy.__version__) < Version('0.20'):
            assert plot.crs.proj4_params['proj'] == proj
        else:
            assert plot.crs.to_dict()['proj'] == proj

    def assert_projection(self, plot, proj):
        opts = hv.Store.lookup_options('bokeh', plot, 'plot')
        assert opts.kwargs['projection'].proj4_params['proj'] == proj