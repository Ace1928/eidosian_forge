from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def trailing_optional(xs):
    if not xs:
        return ''
    return xs[0] + optional(trailing_optional(xs[1:]))