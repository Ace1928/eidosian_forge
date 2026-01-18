import re
import warnings
from typing import Any, Optional
from pyproj._compat import cstrencode
from pyproj._transformer import Factors
from pyproj.crs import CRS
from pyproj.enums import TransformDirection
from pyproj.list import get_proj_operations_map
from pyproj.transformer import Transformer, TransformerFromPipeline
from pyproj.utils import _convertback, _copytobuffer
def to_latlong_def(self) -> Optional[str]:
    """return the definition string of the geographic (lat/lon)
        coordinate version of the current projection"""
    return self.crs.geodetic_crs.to_proj4(4) if self.crs.geodetic_crs else None