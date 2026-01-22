import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def _check_metadata_gs(gs, name='geometry', crs=crs_wgs):
    assert gs.name == name
    assert gs.crs == crs