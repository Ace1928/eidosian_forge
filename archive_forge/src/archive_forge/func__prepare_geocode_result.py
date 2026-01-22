from collections import defaultdict
import time
import pandas as pd
from shapely.geometry import Point
import geopandas
def _prepare_geocode_result(results):
    """
    Helper function for the geocode function

    Takes a dict where keys are index entries, values are tuples containing:
    (address, (lat, lon))

    """
    d = defaultdict(list)
    index = []
    for i, s in results.items():
        if s is None:
            p = Point()
            address = None
        else:
            address, loc = s
            if loc is None:
                p = Point()
            else:
                p = Point(loc[1], loc[0])
        d['geometry'].append(p)
        d['address'].append(address)
        index.append(i)
    df = geopandas.GeoDataFrame(d, index=index, crs='EPSG:4326')
    return df