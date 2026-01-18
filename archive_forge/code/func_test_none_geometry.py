import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_none_geometry(self):
    df = self.nybb.copy()
    df.loc[0, df.geometry.name] = None
    m = df.explore()
    self._fetch_map_string(m)