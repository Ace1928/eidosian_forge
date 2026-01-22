from statistics import mean
import geopandas
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from packaging.version import Version
def _style_default(x):
    return {**style_kwds, **style_kwds_function(x)}