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
def _safe_pj_transform(src_crs, tgt_crs, x, y, z=None, trap=True):
    transformer = _get_transformer_from_crs(src_crs, tgt_crs)
    if z is None:
        z = np.zeros_like(x)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Conversion of an array with ndim > 0')
        return transformer.transform(x, y, z, errcheck=trap)