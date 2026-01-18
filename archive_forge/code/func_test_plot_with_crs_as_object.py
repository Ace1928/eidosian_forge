import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_crs_as_object(self):
    plot = self.da.hvplot.image('x', 'y', crs=self.crs)
    self.assertCRS(plot)