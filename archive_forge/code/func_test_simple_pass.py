import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_simple_pass(self):
    """Make sure default pass"""
    self.nybb.explore()
    self.world.explore()
    self.cities.explore()
    self.world.geometry.explore()