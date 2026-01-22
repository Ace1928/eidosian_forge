from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
def _as_ragged_element_array(self):
    return np.array([_RaggedElement.ragged_or_nan(self[i]) for i in range(len(self))])