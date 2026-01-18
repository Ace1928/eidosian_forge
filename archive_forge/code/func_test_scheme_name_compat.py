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
@pytest.mark.parametrize('scheme', ['FISHER_JENKS', 'FISHERJENKS'])
def test_scheme_name_compat(self, scheme):
    ax = self.df.plot(column='NEGATIVES', scheme=scheme, k=3, legend=True)
    assert len(ax.get_legend().get_texts()) == 3