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
special method that allows pyproj.Proj instance to be pickled