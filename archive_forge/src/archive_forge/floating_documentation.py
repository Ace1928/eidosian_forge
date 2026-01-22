from __future__ import annotations
from typing import ClassVar
import numpy as np
from pandas.core.dtypes.base import register_extension_dtype
from pandas.core.dtypes.common import is_float_dtype
from pandas.core.arrays.numeric import (

        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        