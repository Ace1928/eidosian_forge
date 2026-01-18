import itertools
from packaging.version import Version
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import (
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.datasets import get_path
import geopandas._compat as compat
from geopandas.plotting import GeoplotAccessor
import pytest
import matplotlib.pyplot as plt
@pytest.mark.skip(reason='array-like style_kwds not supported for mixed geometry types (#1379)')
def test_style_kwargs_linestyle_listlike(self):
    ls = ['solid', 'dotted', 'dashdot']
    exp_ls = [_style_to_linestring_onoffseq(style, 1) for style in ls]
    for ax in [self.series.plot(linestyle=ls, linewidth=1), self.series.plot(linestyles=ls, linewidth=1), self.df.plot(linestyles=ls, linewidth=1)]:
        assert exp_ls == ax.collections[0].get_linestyle()