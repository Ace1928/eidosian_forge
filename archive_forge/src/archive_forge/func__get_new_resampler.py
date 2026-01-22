from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import numpy as np
import pandas
import pandas.core.resample
from pandas._libs import lib
from pandas.core.dtypes.common import is_list_like
from modin.logging import ClassLogger
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import _inherit_docstrings
def _get_new_resampler(key):
    subset = self._dataframe[key]
    resampler = type(self)(subset, **self.resample_kwargs)
    return resampler