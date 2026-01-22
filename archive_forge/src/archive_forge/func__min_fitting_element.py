from __future__ import annotations
from collections.abc import (
from datetime import timedelta
import operator
from sys import getsizeof
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.algos import unique_deltas
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCTimedeltaIndex
from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.ops.common import unpack_zerodim_and_defer
def _min_fitting_element(self, lower_limit: int) -> int:
    """Returns the smallest element greater than or equal to the limit"""
    no_steps = -(-(lower_limit - self.start) // abs(self.step))
    return self.start + abs(self.step) * no_steps