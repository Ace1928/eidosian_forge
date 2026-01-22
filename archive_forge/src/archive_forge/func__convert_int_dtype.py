from __future__ import annotations
from functools import partial
import operator
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas.compat import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import (
from pandas.core.ops import invalid_comparison
from pandas.core.strings.object_array import ObjectStringArrayMixin
def _convert_int_dtype(self, result):
    if isinstance(result, pa.Array):
        result = result.to_numpy(zero_copy_only=False)
    else:
        result = result.to_numpy()
    if result.dtype == np.int32:
        result = result.astype(np.int64)
    return result