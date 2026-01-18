import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_geo_as_true_crs_no_crs_on_data_returns_default(self):
    da = self.da.copy()
    da.rio._crs = False
    da.attrs = {'bar': self.crs}
    plot = da.hvplot.image('x', 'y', geo=True)
    self.assertCRS(plot, 'eqc')