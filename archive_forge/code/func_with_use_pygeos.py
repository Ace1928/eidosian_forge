from contextlib import contextmanager
import glob
import os
import pathlib
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from geopandas import _compat as compat
import geopandas
from shapely.geometry import Point
@contextmanager
def with_use_pygeos(option):
    orig = geopandas.options.use_pygeos
    geopandas.options.use_pygeos = option
    try:
        yield
    finally:
        geopandas.options.use_pygeos = orig