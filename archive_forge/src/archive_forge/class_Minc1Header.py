from __future__ import annotations
from numbers import Integral
import numpy as np
from .externals.netcdf import netcdf_file
from .fileslice import canonical_slicers
from .spatialimages import SpatialHeader, SpatialImage
class Minc1Header(MincHeader):

    @classmethod
    def may_contain_header(klass, binaryblock):
        return binaryblock[:4] == b'CDF\x01'