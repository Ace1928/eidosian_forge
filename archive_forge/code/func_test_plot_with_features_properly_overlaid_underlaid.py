import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_features_properly_overlaid_underlaid(self):
    plot = self.df.hvplot.points('x', 'y', features=['land', 'borders'])
    assert plot.get(0).group == 'Land'
    assert plot.get(2).group == 'Borders'