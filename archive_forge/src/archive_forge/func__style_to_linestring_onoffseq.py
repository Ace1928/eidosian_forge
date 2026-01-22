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
def _style_to_linestring_onoffseq(linestyle, linewidth):
    """Converts a linestyle string representation, namely one of:
        ['dashed',  'dotted', 'dashdot', 'solid'],
    documented in `Collections.set_linestyle`,
    to the form `onoffseq`.
    """
    offset, dashes = matplotlib.lines._get_dash_pattern(linestyle)
    return matplotlib.lines._scale_dashes(offset, dashes, linewidth)