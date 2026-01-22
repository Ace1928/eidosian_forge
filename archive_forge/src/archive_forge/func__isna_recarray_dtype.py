from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def _isna_recarray_dtype(values: np.rec.recarray, inf_as_na: bool) -> npt.NDArray[np.bool_]:
    result = np.zeros(values.shape, dtype=bool)
    for i, record in enumerate(values):
        record_as_array = np.array(record.tolist())
        does_record_contain_nan = isna_all(record_as_array)
        does_record_contain_inf = False
        if inf_as_na:
            does_record_contain_inf = bool(_has_record_inf_value(record_as_array))
        result[i] = np.any(np.logical_or(does_record_contain_nan, does_record_contain_inf))
    return result