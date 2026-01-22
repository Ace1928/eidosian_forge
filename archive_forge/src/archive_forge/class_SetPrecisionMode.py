import warnings
from enum import IntEnum
import numpy as np
from shapely import _geometry_helpers, geos_version, lib
from shapely._enum import ParamEnum
from shapely.decorators import multithreading_enabled, requires_geos
class SetPrecisionMode(ParamEnum):
    valid_output = 0
    pointwise = 1
    keep_collapsed = 2