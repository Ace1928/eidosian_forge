from __future__ import annotations
import numbers
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import (
from pandas.core.arrays.masked import (

        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        