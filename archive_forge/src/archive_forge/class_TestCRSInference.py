import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
class TestCRSInference(TestGeo):

    def setUp(self):
        if sys.platform == 'win32':
            raise SkipTest('Skip CRS inference on Windows')
        super().setUp()

    def test_plot_with_crs_as_proj_string(self):
        da = self.da.copy()
        da.rio._crs = False
        plot = self.da.hvplot.image('x', 'y', crs='epsg:32618')
        self.assertCRS(plot)

    def test_plot_with_geo_as_true_crs_undefined(self):
        plot = self.da.hvplot.image('x', 'y', geo=True)
        self.assertCRS(plot)