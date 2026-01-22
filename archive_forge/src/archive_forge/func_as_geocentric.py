from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def as_geocentric(self):
    """
        Return a new Geocentric CRS with the same ellipse/datum as this
        CRS.

        """
    return CRS({'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'GeodeticCRS', 'name': 'unknown', 'datum': self.datum.to_json_dict(), 'coordinate_system': {'subtype': 'Cartesian', 'axis': [{'name': 'Geocentric X', 'abbreviation': 'X', 'direction': 'geocentricX', 'unit': 'metre'}, {'name': 'Geocentric Y', 'abbreviation': 'Y', 'direction': 'geocentricY', 'unit': 'metre'}, {'name': 'Geocentric Z', 'abbreviation': 'Z', 'direction': 'geocentricZ', 'unit': 'metre'}]}})