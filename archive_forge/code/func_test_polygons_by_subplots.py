import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_polygons_by_subplots(self):
    polygons = self.polygons.hvplot(geo=True, by='name', subplots=True)
    assert isinstance(polygons, hv.core.layout.NdLayout)