from statistics import mean
import geopandas
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from packaging.version import Version
def _colormap_helper(_cmap, n_resample=None, idx=None):
    """Helper for MPL deprecation - GH#2596"""
    if not n_resample:
        return cm.get_cmap(_cmap)
    elif MPL_361:
        return cm.get_cmap(_cmap).resampled(n_resample)(idx)
    else:
        return cm.get_cmap(_cmap, n_resample)(idx)