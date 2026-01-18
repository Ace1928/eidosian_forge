import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def total_area(s):
    return s.area.sum()