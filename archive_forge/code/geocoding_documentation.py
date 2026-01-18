from collections import defaultdict
import time
import pandas as pd
from shapely.geometry import Point
import geopandas

    Helper function for the geocode function

    Takes a dict where keys are index entries, values are tuples containing:
    (address, (lat, lon))

    