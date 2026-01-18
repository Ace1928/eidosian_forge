from __future__ import annotations
import warnings
import numpy as np
from xarray.core import dtypes, duck_array_ops, nputils, utils
from xarray.core.duck_array_ops import (
In house nanmean. ddof argument will be used in _nanvar method