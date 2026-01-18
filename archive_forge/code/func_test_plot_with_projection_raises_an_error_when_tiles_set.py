import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_plot_with_projection_raises_an_error_when_tiles_set(self):
    da = self.da.copy()
    with self.assertRaisesRegex(ValueError, 'Tiles can only be used with output projection'):
        da.hvplot.image('x', 'y', crs=self.crs, projection='Robinson', tiles=True)