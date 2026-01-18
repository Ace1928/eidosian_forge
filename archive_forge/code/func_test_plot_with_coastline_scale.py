import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_coastline_scale(self):
    plot = self.df.hvplot.points('x', 'y', geo=True, coastline='10m')
    opts = plot.get(1).opts.get('plot')
    assert opts.kwargs['scale'] == '10m'